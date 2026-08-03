"""Radar Ghost Dataset CMTO label decoding.

Background is intentionally not treated as a clean negative.  It means that
no detailed annotation was assigned, so using it as "real" would introduce
label noise.  The binary target convention used here is 0=real, 1=multipath,
-1=ignore.
"""

from dataclasses import dataclass

import numpy as np


CLASS_NAMES = {
    1: "pedestrian",
    2: "cyclist",
    3: "car",
    4: "large_vehicle",
    5: "motorcycle",
}
VALID_BOUNCE_TYPES = frozenset((0, 1, 2, 3))
VALID_BOUNCE_ORDERS = frozenset((0, 1, 2, 3, 4, 6))


@dataclass(frozen=True)
class DecodedGhostLabel:
    label_id: int
    class_id: int = -1
    is_main: int = -1
    bounce_type: int = -1
    bounce_order: int = -1
    binary_target: int = -1
    sketchy: bool = False
    special: str = ""


def decode_cmto_label(label_id, include_sketchy=False, include_undecided=True):
    """Decode one official label and assign the binary research target."""

    value = int(label_id)
    if value == 0:
        return DecodedGhostLabel(value, special="background")
    if value == -1:
        return DecodedGhostLabel(value, special="ignore")
    if value == -2:
        return DecodedGhostLabel(value, special="noise")

    sketchy = value < -2
    encoded = abs(value)
    if not 1000 <= encoded <= 9999:
        return DecodedGhostLabel(value, sketchy=sketchy, special="invalid")

    class_id = (encoded // 1000) % 10
    is_main = (encoded // 100) % 10
    bounce_type = (encoded // 10) % 10
    bounce_order = encoded % 10
    valid = (
        class_id in CLASS_NAMES
        and is_main in (0, 1)
        and bounce_type in VALID_BOUNCE_TYPES
        and bounce_order in VALID_BOUNCE_ORDERS
    )
    if not valid:
        target = -1
        special = "invalid"
    elif sketchy and not include_sketchy:
        target = -1
        special = "sketchy"
    elif bounce_order == 1:
        target = 0
        special = ""
    elif bounce_order == 0 and not include_undecided:
        target = -1
        special = "undecided"
    else:
        # Official non-main multipath labels may end in 00.  These are still
        # multipath when include_undecided is enabled.
        target = 1
        special = ""
    return DecodedGhostLabel(
        label_id=value,
        class_id=class_id,
        is_main=is_main,
        bounce_type=bounce_type,
        bounce_order=bounce_order,
        binary_target=target,
        sketchy=sketchy,
        special=special,
    )


def label_id_to_binary_target(
    label_ids,
    include_sketchy=False,
    include_undecided=True,
):
    """Vectorized CMTO-to-binary mapping with -1 for ignored detections."""

    values = np.asarray(label_ids)
    flat = values.reshape(-1)
    targets = np.empty(flat.shape, dtype=np.int8)
    for index, value in enumerate(flat):
        targets[index] = decode_cmto_label(
            value,
            include_sketchy=include_sketchy,
            include_undecided=include_undecided,
        ).binary_target
    return targets.reshape(values.shape)


def decode_label_arrays(label_ids, include_sketchy=False, include_undecided=True):
    """Return binary and decomposed CMTO arrays for prepared artifacts."""

    values = np.asarray(label_ids).reshape(-1)
    target = np.empty(values.shape, dtype=np.int8)
    class_id = np.empty(values.shape, dtype=np.int8)
    is_main = np.empty(values.shape, dtype=np.int8)
    bounce_type = np.empty(values.shape, dtype=np.int8)
    bounce_order = np.empty(values.shape, dtype=np.int8)
    sketchy = np.empty(values.shape, dtype=np.bool_)
    for index, value in enumerate(values):
        decoded = decode_cmto_label(
            value,
            include_sketchy=include_sketchy,
            include_undecided=include_undecided,
        )
        target[index] = decoded.binary_target
        class_id[index] = decoded.class_id
        is_main[index] = decoded.is_main
        bounce_type[index] = decoded.bounce_type
        bounce_order[index] = decoded.bounce_order
        sketchy[index] = decoded.sketchy
    return {
        "target": target,
        "class_id": class_id,
        "is_main": is_main,
        "bounce_type": bounce_type,
        "bounce_order": bounce_order,
        "sketchy": sketchy,
    }
