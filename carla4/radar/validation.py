"""Dependency-light accuracy aggregation for the CARLA radar validator."""

from dataclasses import dataclass, field
import math


def _percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def error_statistics(values):
    """Summarize signed errors without requiring pandas or SciPy."""

    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "bias": None,
            "mae": None,
            "rmse": None,
            "median_abs": None,
            "p95_abs": None,
            "max_abs": None,
        }
    absolute = sorted(abs(value) for value in finite)
    return {
        "count": len(finite),
        "bias": sum(finite) / len(finite),
        "mae": sum(absolute) / len(absolute),
        "rmse": math.sqrt(
            sum(value * value for value in finite) / len(finite)
        ),
        "median_abs": _percentile(absolute, 0.50),
        "p95_abs": _percentile(absolute, 0.95),
        "max_abs": absolute[-1],
    }


@dataclass
class BackendAccuracy:
    """Accumulate target availability, identity, synchronization, and error."""

    name: str
    identity_available: bool
    samples: int = 0
    observable_frames: int = 0
    reported_frames: int = 0
    reported_when_observable: int = 0
    missed_when_observable: int = 0
    correct_target_frames: int = 0
    wrong_target_frames: int = 0
    callback_error_frames: int = 0
    unsynchronized_frames: int = 0
    range_errors_current: list = field(default_factory=list)
    velocity_errors_current: list = field(default_factory=list)
    range_errors_aligned: list = field(default_factory=list)
    velocity_errors_aligned: list = field(default_factory=list)
    correct_range_errors: list = field(default_factory=list)
    correct_velocity_errors: list = field(default_factory=list)
    sensor_frame_lags: list = field(default_factory=list)

    def update(
        self,
        *,
        observable,
        reported,
        target_id,
        lead_id,
        synchronized,
        callback_error,
        frame_lag,
        range_error_current=None,
        velocity_error_current=None,
        range_error_aligned=None,
        velocity_error_aligned=None,
    ):
        self.samples += 1
        self.observable_frames += int(bool(observable))
        self.reported_frames += int(bool(reported))
        if observable and reported:
            self.reported_when_observable += 1
        if observable and not reported:
            self.missed_when_observable += 1
        if not synchronized:
            self.unsynchronized_frames += 1
        if callback_error:
            self.callback_error_frames += 1
        if frame_lag is not None and math.isfinite(float(frame_lag)):
            self.sensor_frame_lags.append(float(frame_lag))

        correct = None
        if self.identity_available and reported:
            correct = int(target_id) == int(lead_id)
            if correct:
                self.correct_target_frames += 1
            else:
                self.wrong_target_frames += 1

        def append_if_finite(destination, value):
            if value is not None and math.isfinite(float(value)):
                destination.append(float(value))

        if observable and reported:
            append_if_finite(
                self.range_errors_current,
                range_error_current,
            )
            append_if_finite(
                self.velocity_errors_current,
                velocity_error_current,
            )
            append_if_finite(
                self.range_errors_aligned,
                range_error_aligned,
            )
            append_if_finite(
                self.velocity_errors_aligned,
                velocity_error_aligned,
            )
            if correct:
                append_if_finite(
                    self.correct_range_errors,
                    range_error_aligned,
                )
                append_if_finite(
                    self.correct_velocity_errors,
                    velocity_error_aligned,
                )
        return correct

    @staticmethod
    def _ratio(numerator, denominator):
        if denominator <= 0:
            return None
        return numerator / denominator

    def summary(self):
        """Return a JSON-serializable backend summary."""

        identified = self.correct_target_frames + self.wrong_target_frames
        return {
            "samples": self.samples,
            "lead_observable_frames": self.observable_frames,
            "reported_frames": self.reported_frames,
            "detection_rate_when_observable": self._ratio(
                self.reported_when_observable,
                self.observable_frames,
            ),
            "miss_rate_when_observable": self._ratio(
                self.missed_when_observable,
                self.observable_frames,
            ),
            "identity_available": self.identity_available,
            "identified_output_frames": identified,
            "correct_target_frames": self.correct_target_frames,
            "wrong_target_frames": self.wrong_target_frames,
            "correct_target_rate": self._ratio(
                self.correct_target_frames,
                identified,
            ),
            "callback_error_frames": self.callback_error_frames,
            "unsynchronized_frames": self.unsynchronized_frames,
            "sensor_frame_lag": error_statistics(self.sensor_frame_lags),
            "selected_output_range_error_current_m": error_statistics(
                self.range_errors_current
            ),
            "selected_output_velocity_error_current_mps": error_statistics(
                self.velocity_errors_current
            ),
            "selected_output_range_error_latency_aligned_m": (
                error_statistics(self.range_errors_aligned)
            ),
            "selected_output_velocity_error_latency_aligned_mps": (
                error_statistics(self.velocity_errors_aligned)
            ),
            "correct_target_range_error_latency_aligned_m": (
                error_statistics(self.correct_range_errors)
            ),
            "correct_target_velocity_error_latency_aligned_mps": (
                error_statistics(self.correct_velocity_errors)
            ),
        }
