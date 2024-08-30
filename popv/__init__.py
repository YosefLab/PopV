"""PopV."""

# Set default logging handler to avoid logging with logging.lastResort logger.
import logging

import scanpy as sc

from ._settings import settings
from . import algorithms, annotation, preprocessing, visualization

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata
package_name = "popv"
__version__ = importlib_metadata.version(package_name)

settings.verbosity = logging.INFO

# Jax sets the root logger, this prevents double output.
popv_logger = logging.getLogger("popv")
popv_logger.propagate = False


__all__ = [
    "settings",
    "algorithms",
    "annotation",
    "preprocessing",
    "visualization"
]
