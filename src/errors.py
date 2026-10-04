"""Exception types with user-presentable messages.

The UI catches ``FashionAIError`` subclasses and shows ``str(exc)`` directly,
so messages should say what went wrong and what the user can do about it.
"""

from __future__ import annotations


class FashionAIError(Exception):
    """Base class for expected, user-facing failures."""


class InvalidImageError(FashionAIError):
    """The supplied file is missing, unsupported or not a decodable image."""


class DetectorUnavailableError(FashionAIError):
    """The clothing detector could not be loaded (missing package or weights)."""


class DetectionError(FashionAIError):
    """The detector was loaded but inference on an image failed."""


class EmbeddingError(FashionAIError):
    """CLIP could not be loaded or produced an invalid embedding."""


class WardrobeStoreError(FashionAIError):
    """The wardrobe database is unreadable or an operation on it failed."""


class CheckpointError(FashionAIError):
    """A model checkpoint is missing, corrupt or incompatible."""


class DatasetMissingError(FashionAIError):
    """A training/evaluation dataset is missing or malformed."""
