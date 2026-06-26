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

# COCO classes for obstacle detection (vehicles + pedestrians)
COCO_OBSTACLE_CLASSES = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck
OBSTACLE_LABELS = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Approximate real-world heights (metres) for monocular distance estimation
REFERENCE_HEIGHTS = {
    0: 1.7,   # person
    2: 1.5,   # car
    3: 1.2,   # motorcycle
    5: 3.2,   # bus
    7: 2.8,   # truck
}

MAX_VISION_RANGE = 50.0


def empty_obstacle_features():
    """Return default obstacle features when nothing is detected."""
    return {
        "has_detection": False,
        "vision_distance": MAX_VISION_RANGE,
        "vision_confidence": 0.0,
        "vision_bbox_height": 0.0,
        "vision_bbox_area": 0.0,
        "vision_bbox_center_x": 0.5,
        "vision_bbox": None,
        "vision_obstacle_class": -1,
        "vision_obstacle_label": "none",
        "obstacle_boxes": [],
    }


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

    def _infer_scene(self, frame):
        if frame is None:
            return None

        return self.model(
            frame,
            conf=self.confidence,
            classes=sorted(set([COCO_TRAFFIC_LIGHT] + COCO_OBSTACLE_CLASSES)),
            verbose=False,
            device=self.device,
        )

    @staticmethod
    def _clip_box(box, frame_w, frame_h):
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, min(frame_w - 1, x1))
        x2 = max(0, min(frame_w, x2))
        y1 = max(0, min(frame_h - 1, y1))
        y2 = max(0, min(frame_h, y2))
        return x1, y1, x2, y2

    def _extract_light_from_detections(self, frame, xyxy, confidences, class_ids):
        light_indices = [idx for idx, class_id in enumerate(class_ids) if class_id == COCO_TRAFFIC_LIGHT]
        if not light_indices:
            return empty_visual_features()

        frame_h, frame_w = frame.shape[:2]
        frame_area = float(max(1, frame_h * frame_w))

        def box_area(index):
            x1, y1, x2, y2 = xyxy[index]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            return (width * height) / frame_area

        best_idx = max(light_indices, key=box_area)
        best_box = self._clip_box(xyxy[best_idx], frame_w, frame_h)
        best_area = float(box_area(best_idx))
        best_conf = float(confidences[best_idx])

        x1, y1, x2, y2 = best_box
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

    def _extract_obstacle_from_detections(self, frame, xyxy, confidences, class_ids, focal_length_y):
        obstacle = empty_obstacle_features()
        frame_h, frame_w = frame.shape[:2]
        frame_area = float(max(1, frame_h * frame_w))
        detections = []
        primary_idx = -1
        best_distance = MAX_VISION_RANGE

        for idx, class_id in enumerate(class_ids):
            class_id = int(class_id)
            if class_id not in COCO_OBSTACLE_CLASSES:
                continue

            x1, y1, x2, y2 = self._clip_box(xyxy[idx], frame_w, frame_h)
            width = max(1.0, float(x2 - x1))
            height = max(1.0, float(y2 - y1))
            center_x = ((x1 + x2) * 0.5) / max(1.0, float(frame_w))
            area = (width * height) / frame_area
            ref_height = REFERENCE_HEIGHTS.get(class_id, 1.5)
            est_dist = min(focal_length_y * ref_height / height, MAX_VISION_RANGE)

            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "class_id": class_id,
                    "label": OBSTACLE_LABELS.get(class_id, str(class_id)),
                    "confidence": float(confidences[idx]),
                    "distance": round(est_dist, 4),
                    "center_x": float(center_x),
                    "bbox_height": round(height / frame_h, 4),
                    "bbox_area": round(area, 6),
                    "is_primary": False,
                }
            )

            if 0.325 <= center_x <= 0.675 and est_dist < best_distance and area > 0.002:
                best_distance = est_dist
                primary_idx = len(detections) - 1

        if primary_idx >= 0:
            detections[primary_idx]["is_primary"] = True
            chosen = detections[primary_idx]
            obstacle.update(
                {
                    "has_detection": True,
                    "vision_distance": chosen["distance"],
                    "vision_confidence": chosen["confidence"],
                    "vision_bbox_height": chosen["bbox_height"],
                    "vision_bbox_area": chosen["bbox_area"],
                    "vision_bbox_center_x": chosen["center_x"],
                    "vision_bbox": chosen["bbox"],
                    "vision_obstacle_class": chosen["class_id"],
                    "vision_obstacle_label": chosen["label"],
                }
            )

        obstacle["obstacle_boxes"] = detections
        return obstacle

    def extract_scene_features(self, frame, focal_length_y=320.0):
        if frame is None:
            return {
                "visual": empty_visual_features(),
                "obstacle": empty_obstacle_features(),
            }

        results = self._infer_scene(frame)
        if not results or len(results[0].boxes) == 0:
            return {
                "visual": empty_visual_features(),
                "obstacle": empty_obstacle_features(),
            }

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        return {
            "visual": self._extract_light_from_detections(frame, xyxy, confidences, class_ids),
            "obstacle": self._extract_obstacle_from_detections(
                frame, xyxy, confidences, class_ids, focal_length_y
            ),
        }

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
        return self.extract_scene_features(frame)["visual"]

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

    def extract_obstacle_features(self, frame, focal_length_y=320.0):
        """
        Detect nearest vehicle or pedestrian and estimate distance from
        bounding-box height using pinhole-camera geometry.

        Distance ≈ focal_length_y × reference_height / bbox_height_pixels

        For the default CameraManager (640×480, FOV=90°):
            focal_length_y = (480 / 2) / tan(vertical_fov / 2)
                           = 240 / 0.75 ≈ 320 pixels

        Only considers detections in the centre 50 % of the frame width
        to filter out adjacent-lane traffic.

        Returns the same dict shape as empty_obstacle_features().
        """
        return self.extract_scene_features(frame, focal_length_y=focal_length_y)["obstacle"]

    def get_state_name(self, state):
        return TL_STATE_NAMES.get(state, "unknown")


class VisionDistanceTracker:
    """
    Wraps YOLO obstacle detection and smooths the raw distance estimate
    with an exponential moving average to produce stable relative-velocity
    readings from frame-to-frame distance changes.

    Returns dicts with keys: distance, relative_velocity, obstacle_speed
    — matching the FrontRadar.get() interface so callers can swap freely.
    """

    DEFAULT_FPS = 20
    DEFAULT_ALPHA = 0.4
    DEFAULT_HOLD_MISSING_FRAMES = 6
    DEFAULT_RELEASE_RATE_MPS = 8.0

    def __init__(
        self,
        yolo=None,
        alpha=None,
        max_range=None,
        fps=None,
        hold_missing_frames=None,
        release_rate_mps=None,
    ):
        self.yolo = yolo
        self.alpha = alpha if alpha is not None else self.DEFAULT_ALPHA
        self.max_range = max_range if max_range is not None else MAX_VISION_RANGE
        self.fps = fps if fps is not None else self.DEFAULT_FPS
        self.hold_missing_frames = (
            hold_missing_frames
            if hold_missing_frames is not None
            else self.DEFAULT_HOLD_MISSING_FRAMES
        )
        self.release_rate_mps = (
            release_rate_mps if release_rate_mps is not None else self.DEFAULT_RELEASE_RATE_MPS
        )
        self._smoothed = self.max_range
        self._raw_prev = self.max_range
        self._smoothed_velocity = 0.0
        self.vel_alpha = 0.25
        self._ego_speed = 0.0
        self._missed_frames = 0

    def update_ego_speed(self, speed):
        self._ego_speed = speed

    def update(self, obstacle_features):
        """Update the tracked obstacle state from already-extracted perception data.

        Velocity is computed from raw (pre-smoothing) distance to reduce lag,
        then smoothed separately with its own EMA. On missed detections, the
        smoothed distance releases gradually and velocity decays toward zero.
        """
        obstacle_features = obstacle_features or empty_obstacle_features()

        if obstacle_features.get("has_detection"):
            raw_distance = float(obstacle_features["vision_distance"])

            # Velocity from raw signal (less lag than from smoothed distance)
            raw_velocity = (self._raw_prev - raw_distance) * self.fps
            raw_velocity = np.clip(raw_velocity, -8.0, 8.0)
            self._raw_prev = raw_distance

            # Smooth distance separately
            self._smoothed = self.alpha * raw_distance + (1.0 - self.alpha) * self._smoothed

            # Smooth velocity separately with its own alpha
            self._smoothed_velocity = (
                self.vel_alpha * raw_velocity
                + (1.0 - self.vel_alpha) * self._smoothed_velocity
            )
            self._missed_frames = 0
        else:
            self._missed_frames += 1
            if self._missed_frames > self.hold_missing_frames:
                release_step = self.release_rate_mps / max(1.0, float(self.fps))
                self._smoothed = min(self.max_range, self._smoothed + release_step)
                self._raw_prev = self._smoothed
            # Decay velocity on missed frames
            self._smoothed_velocity = self._smoothed_velocity * 0.85

        obstacle_speed = max(0.0, self._ego_speed - self._smoothed_velocity)

        return {
            "distance": round(self._smoothed, 4),
            "relative_velocity": round(self._smoothed_velocity, 4),
            "obstacle_speed": round(obstacle_speed, 4),
        }

    def reset(self):
        self._smoothed = self.max_range
        self._raw_prev = self.max_range
        self._smoothed_velocity = 0.0
        self._missed_frames = 0
