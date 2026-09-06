"""The one place the deployed radar is checked against the trained one.

`test_throttle_brake_live.py` and every scenario driver used to carry their
own copy of this gate, and the copies drifted. They now call
`check_radar_provenance` and print what it returns.

Two kinds of difference are treated differently on purpose:

* **Sensor identity** (backend, range, and the config signature, which covers
  noise, resolution, detection, tracker, selector and point emission) must
  match or the run is refused. A model trained on one sensor says nothing
  about another.
* **Ghost injection and ghost filtering** (multipath mode, ghost rate and SNR
  knobs, oracle or learned filter and its threshold) may differ, because that
  difference *is* the experiment: a clean-trained controller deployed against
  ghosts, or behind a filter, is arms A to C of the study. Differences are
  reported as warnings so they are visible in every log, never silent.
"""


SENSOR_KEYS = ("radar_backend", "radar_range_m", "radar_config_signature")
FILTER_KEYS = (
    "radar_ghost_detector_signature",
    "radar_ghost_threshold",
    "radar_ghost_oracle",
)


class RadarProvenanceError(RuntimeError):
    """The deployed sensor is not the one the model was trained on."""


def _same(left, right):
    if isinstance(left, float) or isinstance(right, float):
        try:
            return abs(float(left) - float(right)) <= 1.0e-6
        except (TypeError, ValueError):
            return False
    return left == right


def check_radar_provenance(model_config, runtime_metadata):
    """Raise on a sensor mismatch; return warnings for injection differences.

    ``model_config`` is the trainer's ``model_config.json`` mapping and
    ``runtime_metadata`` the dict from ``describe_radar_configuration`` for
    the sensor about to be built. Both carry the same keys because the trainer
    copies them from ``dataset_config.json``.
    """

    trained_backend = model_config.get("radar_backend", "native")
    runtime_backend = runtime_metadata.get("radar_backend")
    if runtime_backend != trained_backend:
        raise RadarProvenanceError(
            "Sensor distribution mismatch: model data used radar backend "
            f"{trained_backend!r}, runtime requested {runtime_backend!r}. "
            "Recollect/retrain or select the trained backend."
        )
    if trained_backend == "realistic":
        trained_signature = model_config.get("radar_config_signature")
        runtime_signature = runtime_metadata.get("radar_config_signature")
        if trained_signature != runtime_signature:
            raise RadarProvenanceError(
                "Realistic radar configuration mismatch: model data used "
                f"{trained_signature!r}, runtime requested {runtime_signature!r}. "
                "Ghost settings are outside the signature, so this is a real "
                "sensor difference (noise, resolution, tracker or point "
                "emission). Recollect and retrain."
            )

    warnings = []
    trained_injection = model_config.get("radar_ghost_injection") or {}
    runtime_injection = runtime_metadata.get("radar_ghost_injection") or {}
    for key in sorted(set(trained_injection) | set(runtime_injection)):
        if key == "profile_name":
            continue
        trained_value = trained_injection.get(key)
        runtime_value = runtime_injection.get(key)
        if not _same(trained_value, runtime_value):
            warnings.append(
                f"ghost injection differs: {key} trained={trained_value!r} "
                f"runtime={runtime_value!r}"
            )
    for key in FILTER_KEYS:
        trained_value = model_config.get(key)
        runtime_value = runtime_metadata.get(key)
        if key == "radar_ghost_oracle":
            trained_value = bool(trained_value)
            runtime_value = bool(runtime_value)
        if not _same(trained_value, runtime_value):
            warnings.append(
                f"ghost filter differs: {key} trained={trained_value!r} "
                f"runtime={runtime_value!r}"
            )
    return warnings


def print_provenance_warnings(warnings, prefix="  "):
    if not warnings:
        return
    print(f"{prefix}WARNING: deployed radar differs from the training radar in ghost settings.")
    print(f"{prefix}         This is the intended experiment only if you asked for it.")
    for line in warnings:
        print(f"{prefix}         - {line}")
