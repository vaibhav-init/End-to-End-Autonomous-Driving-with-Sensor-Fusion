#!/usr/bin/env python3
"""
YOLO perception helpers for CARLA.

This module keeps the perception side intentionally non-GT. The exported
features are richer than a binary "light visible" flag so downstream models can
learn spatial context from camera observations alone.
"""

import threading

import numpy as np

try:
    import carla
except ImportError:
    pass

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed. Run: pip install ultralytics")


TL_NONE = 0
TL_GREEN = 1
TL_YELLOW = 2
TL_RED = 3

TL_STATE_NAMES = {
    TL_NONE: "none",
    TL_GREEN: "green",
    TL_YELLOW: "yellow",
    TL_RED: "red",
}

COCO_TRAFFIC_LIGHT = 9


def empty_visual_features():
    """Return a zeroed traffic-light feature bundle."""
    return {
        "traffic_light_state": TL_NONE,
        "tl_confidence": 0.0,
        "tl_bbox_area": 0.0,
        "tl_center_x": 0.5,
        "tl_bbox": None,
    }


def classify_traffic_light_color(image_crop):
    """Classify a cropped traffic light patch using HSV heuristics."""
    try:
        import cv2
    except ImportError:
        return TL_NONE

    if image_crop is None or image_crop.size == 0:
        return TL_NONE

    height, width = image_crop.shape[:2]
    if height < 5 or width < 5:
        return TL_NONE

    hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
    bright_mask = hsv[:, :, 2] > 150
    if bright_mask.sum() < 10:
        return TL_NONE

    bright_hsv = hsv[bright_mask]
    hue = bright_hsv[:, 0]

    red_count = np.sum((hue < 10) | (hue > 170))
    yellow_count = np.sum((hue >= 15) & (hue <= 35))
    green_count = np.sum((hue >= 40) & (hue <= 85))

    total = red_count + yellow_count + green_count
    if total == 0:
        return TL_NONE

    max_count = max(red_count, yellow_count, green_count)
    if max_count / total < 0.3:
        return TL_NONE

    if red_count >= yellow_count and red_count >= green_count:
        return TL_RED
    if yellow_count >= red_count and yellow_count >= green_count:
        return TL_YELLOW
    return TL_GREEN


class CameraManager:
    """RGB camera sensor wrapper."""

    def __init__(self, vehicle, world, width=640, height=480, fov=90):
        self._frame = None
        self._lock = threading.Lock()
        self._width = width
        self._height = height

        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        bp.set_attribute("sensor_tick", "0.05")

        transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=-5.0),
        )

        self.sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
        self.sensor.listen(self._on_image)

    def _on_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((self._height, self._width, 4))
        bgr = array[:, :, :3].copy()
        with self._lock:
            self._frame = bgr

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def cleanup(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()


class YOLOPerception:
    """
    Detect traffic lights and expose richer visual features for control.

    The closest candidate is selected by normalized box area rather than raw
    confidence. That gives downstream policy code a stable proxy for proximity.
    """

    def __init__(self, model_name="yolov8n.pt", confidence=0.35, device=None):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

        self.model = YOLO(model_name)
        self.confidence = confidence
        self.device = device
        self._last_state = TL_NONE
        self._same_state_count = 0

    def _stable_state(self, color_state):
        if color_state == self._last_state:
            self._same_state_count += 1
        else:
            self._last_state = color_state
            self._same_state_count = 1

        if self._same_state_count >= 2:
            return color_state
        return TL_NONE

    def extract_light_features(self, frame):
        """
        Return a feature bundle for the largest visible traffic light.

        Keys:
            traffic_light_state: 0..3 from HSV classification
            tl_confidence: YOLO confidence
            tl_bbox_area: normalized bbox area in image coordinates
            tl_center_x: normalized horizontal center in [0, 1]
            tl_bbox: integer pixel bbox tuple or None
        """
        if frame is None:
            return empty_visual_features()

        results = self.model(
            frame,
            conf=self.confidence,
            classes=[COCO_TRAFFIC_LIGHT],
            verbose=False,
            device=self.device,
        )
        if not results or len(results[0].boxes) == 0:
            return empty_visual_features()

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        frame_h, frame_w = frame.shape[:2]
        frame_area = float(max(1, frame_h * frame_w))

        areas = []
        for box in xyxy:
            x1, y1, x2, y2 = box
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            areas.append((width * height) / frame_area)

        best_idx = int(np.argmax(areas))
        best_box = xyxy[best_idx].astype(int)
        best_area = float(areas[best_idx])
        best_conf = float(confidences[best_idx])

        x1, y1, x2, y2 = best_box
        x1 = max(0, min(frame_w - 1, x1))
        x2 = max(0, min(frame_w, x2))
        y1 = max(0, min(frame_h - 1, y1))
        y2 = max(0, min(frame_h, y2))

        crop = frame[y1:y2, x1:x2]
        color_state = classify_traffic_light_color(crop)
        stable_state = self._stable_state(color_state)
        center_x = ((x1 + x2) * 0.5) / max(1.0, float(frame_w))

        return {
            "traffic_light_state": stable_state,
            "tl_confidence": best_conf,
            "tl_bbox_area": best_area,
            "tl_center_x": float(center_x),
            "tl_bbox": (x1, y1, x2, y2),
        }

    def detect_traffic_light(self, frame):
        """
        Backward-compatible adapter returning the legacy tuple.
        """
        features = self.extract_light_features(frame)
        return (
            features["traffic_light_state"],
            features["tl_confidence"],
            features["tl_bbox"],
        )

    def get_state_name(self, state):
        return TL_STATE_NAMES.get(state, "unknown")
