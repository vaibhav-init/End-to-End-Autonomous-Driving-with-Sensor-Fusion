#!/usr/bin/env python3
"""
YOLO Perception Module for CARLA
==================================

Provides real-time traffic light detection and color classification
using YOLOv8 + HSV color analysis.

Components:
    CameraManager  — Attaches an RGB camera to the ego vehicle
    YOLOPerception — Runs YOLOv8n inference + HSV traffic light classifier

Usage:
    from yolo_perception import CameraManager, YOLOPerception

    camera = CameraManager(ego, world)
    yolo = YOLOPerception()

    frame = camera.get_frame()
    if frame is not None:
        tl_state, confidence, bbox = yolo.detect_traffic_light(frame)
        # tl_state: 0=none, 1=green, 2=yellow, 3=red
"""

import numpy as np
import threading

try:
    import carla
except ImportError:
    pass

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  ultralytics not installed. Run: pip install ultralytics")


# Traffic light state encoding
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

# COCO class indices
COCO_TRAFFIC_LIGHT = 9
COCO_STOP_SIGN = 11


# ============================================================================
# HSV Traffic Light Color Classifier
# ============================================================================
def classify_traffic_light_color(image_crop):
    """
    Classify a cropped traffic light image as red, yellow, or green
    using HSV color space analysis.

    Args:
        image_crop: numpy array (H, W, 3) in BGR format

    Returns:
        int: TL_RED=3, TL_YELLOW=2, TL_GREEN=1, or TL_NONE=0 if uncertain
    """
    try:
        import cv2
    except ImportError:
        return TL_NONE

    if image_crop is None or image_crop.size == 0:
        return TL_NONE

    # Ensure minimum size
    h, w = image_crop.shape[:2]
    if h < 5 or w < 5:
        return TL_NONE

    hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)

    # Focus on the brighter pixels (the illuminated light)
    # Value channel > 150 filters out the dark housing
    bright_mask = hsv[:, :, 2] > 150

    if bright_mask.sum() < 10:
        # Not enough bright pixels — might be off or too dark
        return TL_NONE

    bright_hsv = hsv[bright_mask]

    # Count pixels in each color range (Hue channel, 0-180 in OpenCV)
    hue = bright_hsv[:, 0]

    # Red wraps around: H < 10 or H > 170
    red_count = np.sum((hue < 10) | (hue > 170))
    # Yellow: H 15-35
    yellow_count = np.sum((hue >= 15) & (hue <= 35))
    # Green: H 40-85
    green_count = np.sum((hue >= 40) & (hue <= 85))

    total = red_count + yellow_count + green_count
    if total == 0:
        return TL_NONE

    # Require at least 30% dominance for a classification
    max_count = max(red_count, yellow_count, green_count)
    if max_count / total < 0.3:
        return TL_NONE

    if red_count >= yellow_count and red_count >= green_count:
        return TL_RED
    elif yellow_count >= red_count and yellow_count >= green_count:
        return TL_YELLOW
    else:
        return TL_GREEN


# ============================================================================
# Camera Manager — Attaches RGB camera to ego vehicle
# ============================================================================
class CameraManager:
    """Manages an RGB camera sensor attached to the ego vehicle."""

    def __init__(self, vehicle, world, width=640, height=480, fov=90):
        """
        Args:
            vehicle: CARLA ego vehicle actor
            world: CARLA world object
            width: Image width in pixels
            height: Image height in pixels
            fov: Field of view in degrees
        """
        self._frame = None
        self._lock = threading.Lock()
        self._width = width
        self._height = height

        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(width))
        bp.set_attribute('image_size_y', str(height))
        bp.set_attribute('fov', str(fov))
        bp.set_attribute('sensor_tick', '0.05')  # 20 FPS to match sim

        # Mount on top of windshield — high enough to see traffic lights
        transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=-5.0)  # Slight downward tilt
        )

        self.sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
        self.sensor.listen(self._on_image)

    def _on_image(self, image):
        """Callback: convert CARLA image to numpy array (BGR)."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((self._height, self._width, 4))  # BGRA
        bgr = array[:, :, :3].copy()  # Drop alpha, make contiguous

        with self._lock:
            self._frame = bgr

    def get_frame(self):
        """Get the latest camera frame as a BGR numpy array, or None."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def cleanup(self):
        """Destroy the camera sensor."""
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()


# ============================================================================
# YOLO Perception — Traffic light detection + classification
# ============================================================================
class YOLOPerception:
    """
    Runs YOLOv8n to detect traffic lights and classifies their color
    using HSV analysis on the cropped detection.
    """

    def __init__(self, model_name='yolov8n.pt', confidence=0.4, device=None):
        """
        Args:
            model_name: YOLOv8 model file (auto-downloads if not present)
            confidence: Minimum detection confidence threshold
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

        self.model = YOLO(model_name)
        self.confidence = confidence
        self.device = device

        # Track last detection for smoothing
        self._last_state = TL_NONE
        self._same_state_count = 0
        self._last_results = None  # Cache for intersection detection

    def detect_traffic_light(self, frame):
        """
        Detect and classify traffic light in a BGR image frame.

        Args:
            frame: numpy array (H, W, 3) in BGR format

        Returns:
            tuple: (state, confidence, bbox)
                - state: int (0=none, 1=green, 2=yellow, 3=red)
                - confidence: float detection confidence (0.0 if none)
                - bbox: tuple (x1, y1, x2, y2) or None
        """
        if frame is None:
            return TL_NONE, 0.0, None

        # Run YOLO inference — detect traffic lights AND stop signs
        # (stop signs are cached for intersection detection)
        results = self.model(
            frame,
            conf=self.confidence,
            classes=[COCO_TRAFFIC_LIGHT, COCO_STOP_SIGN],
            verbose=False,
            device=self.device,
        )

        # Cache full results for intersection detection
        self._last_results = results

        if not results or len(results[0].boxes) == 0:
            return TL_NONE, 0.0, None

        # Filter for traffic lights only
        boxes = results[0].boxes
        classes = boxes.cls.cpu().numpy()
        tl_mask = classes == COCO_TRAFFIC_LIGHT

        if not tl_mask.any():
            return TL_NONE, 0.0, None

        # Get the highest-confidence traffic light detection
        tl_confidences = boxes.conf.cpu().numpy()[tl_mask]
        tl_boxes = boxes.xyxy.cpu().numpy()[tl_mask]
        best_idx = tl_confidences.argmax()
        best_conf = tl_confidences[best_idx]
        best_box = tl_boxes[best_idx].astype(int)

        x1, y1, x2, y2 = best_box

        # Crop the traffic light region
        crop = frame[
            max(0, y1):min(frame.shape[0], y2),
            max(0, x1):min(frame.shape[1], x2)
        ]

        # Classify color using HSV
        color_state = classify_traffic_light_color(crop)

        # Simple temporal smoothing: require 2 consecutive same-state
        # detections to switch state (reduces flicker)
        if color_state == self._last_state:
            self._same_state_count += 1
        else:
            self._same_state_count = 1
            self._last_state = color_state

        # Only report the state if we've seen it at least twice
        stable_state = color_state if self._same_state_count >= 2 else TL_NONE

        return stable_state, float(best_conf), (x1, y1, x2, y2)

    def detect_intersection(self, frame=None):
        """
        Detect if the vehicle is approaching an intersection using visual cues.

        Returns 1 if a traffic light or stop sign is visible in the frame
        (visual indicator of intersection), 0 otherwise.

        This method uses the cached results from the last detect_traffic_light()
        call. If frame is provided and no cached results exist, runs a fresh
        detection.

        Args:
            frame: optional BGR frame (only used if no cached results)

        Returns:
            int: 1 if approaching intersection, 0 otherwise
        """
        # Use cached results from detect_traffic_light() if available
        results = self._last_results

        if results is None and frame is not None:
            # Run fresh detection
            results = self.model(
                frame,
                conf=self.confidence,
                classes=[COCO_TRAFFIC_LIGHT, COCO_STOP_SIGN],
                verbose=False,
                device=self.device,
            )

        if results is None or len(results[0].boxes) == 0:
            return 0

        # If ANY traffic light or stop sign is detected → approaching intersection
        return 1

    def get_state_name(self, state):
        """Convert state int to human-readable string."""
        return TL_STATE_NAMES.get(state, "unknown")


# ============================================================================
# Junction Proximity Detector (uses CARLA waypoint API)
# ============================================================================
def is_approaching_intersection(ego, carla_map, lookahead_m=30.0, step_m=5.0):
    """
    Check if the ego vehicle is approaching a junction within lookahead_m.

    Uses CARLA's waypoint API — works during both data collection and
    live inference (no YOLO needed for this feature).

    Args:
        ego: CARLA ego vehicle actor
        carla_map: CARLA map object
        lookahead_m: How far ahead to check (meters)
        step_m: Step size for waypoint sampling (meters)

    Returns:
        int: 1 if approaching junction, 0 otherwise
    """
    try:
        wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True)
        if wp is None:
            return 0

        # Check if we're already in a junction
        if wp.is_junction:
            return 1

        # Check waypoints ahead
        dist = step_m
        while dist <= lookahead_m:
            next_wps = wp.next(dist)
            if next_wps and next_wps[0].is_junction:
                return 1
            dist += step_m

        return 0
    except Exception:
        return 0


# ============================================================================
# CARLA Ground Truth Traffic Light State (for data collection only)
# ============================================================================
def get_traffic_light_state_gt(ego):
    """
    Get traffic light state using CARLA's built-in API (ground truth).

    Use this during data collection for perfect labels.
    At inference time, use YOLOPerception instead.

    Args:
        ego: CARLA ego vehicle actor

    Returns:
        int: 0=none, 1=green, 2=yellow, 3=red
    """
    try:
        if ego.is_at_traffic_light():
            tl = ego.get_traffic_light()
            if tl is not None:
                state = tl.get_state()
                state_map = {
                    carla.TrafficLightState.Green: TL_GREEN,
                    carla.TrafficLightState.Yellow: TL_YELLOW,
                    carla.TrafficLightState.Red: TL_RED,
                }
                return state_map.get(state, TL_NONE)
        return TL_NONE
    except Exception:
        return TL_NONE
