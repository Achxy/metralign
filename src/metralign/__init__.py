"""Public Python API for Metralign.

The historical :mod:`drift_sense` namespace remains available so archived
experiments and existing integrations continue to import unchanged.
"""

from drift_sense import __version__
from drift_sense.localizer import LocalizationConfig, Prediction, localize

__all__ = ["LocalizationConfig", "Prediction", "__version__", "localize"]
