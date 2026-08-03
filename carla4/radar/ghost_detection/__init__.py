"""Training and runtime tools for real/synthetic radar ghost detection."""

from .features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from .labels import decode_cmto_label, label_id_to_binary_target

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "decode_cmto_label",
    "label_id_to_binary_target",
]
