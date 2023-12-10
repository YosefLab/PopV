"""PopV."""

# Set default logging handler to avoid logging with logging.lastResort logger.
import logging

import scanpy as sc

from . import algorithms, annotation, preprocessing, visualization
from ._settings import Config

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata
package_name = "popv"
__version__ = importlib_metadata.version(package_name)

Config.verbosity = logging.INFO
Config.num_threads = 10
sc.settings.n_jobs = Config.num_threads

test_var = "test"
popv_logger = logging.getLogger("popv")
popv_logger.propagate = False

__all__ = ["settings", "algorithms", "annotation", "preprocessing", "visualization"]
