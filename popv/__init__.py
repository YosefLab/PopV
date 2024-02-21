"""PopV."""
# Set default logging handler to avoid logging with logging.lastResort logger.
import logging

import scanpy as sc

from ._settings import settings

# this import needs to come after prior imports to prevent circular import
from . import algorithms, annotation, preprocessing, visualization

from importlib.metadata import version

package_name = "popv"
__version__ = version(package_name)

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
