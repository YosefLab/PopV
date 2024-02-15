import logging
from pathlib import Path
from typing import Union, Optional

from rich.console import Console
from rich.logging import RichHandler

import scanpy as sc
import scvi

popv_logger = logging.getLogger("popv")


class Config:
    """
    Config manager for PopV.

    Examples
    --------
    To set the seed

    >>> popv.settings.seed = 1

    To set the verbosity

    >>> import logging
    >>> popv.settings.verbosity = logging.INFO

    To set the number of jobs to be used

    >>> popv.settings.n_jobs = 2

    To set the number of largest dense dataset to be used

    >>> popv.settings.shard_size = 200000

    To enable cuml for rapid GPU based methods

    >>> popv.settings.cuml = True

    """

    def __init__(
        self,
        verbosity: int = logging.WARNING,
        seed: Optional[int] = None,
        logging_dir: str = "./popv_log/",
        n_jobs: int = 1,
        cuml: bool = False,
        shard_size: int = 100000
    ):
        """Set up Config manager for PopV."""
        self.seed = seed
        self.logging_dir = logging_dir
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.cuml = cuml
        self.shard_size = shard_size

    @property
    def logging_dir(self) -> Path:
        """Directory for training logs (default `'./popv_log/'`)."""
        return self._logging_dir

    @logging_dir.setter
    def logging_dir(self, logging_dir: Union[str, Path]):
        self._logging_dir = Path(logging_dir).resolve()

    @property
    def n_jobs(self) -> int:
        """Jobs used for multiprocessing."""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs: int):
        """Random seed for torch and numpy."""
        sc.settings.n_jobs = n_jobs
        self._n_jobs = n_jobs

    @property
    def cuml(self) -> int:
        """Use RAPIDS and cuml."""
        return self._cuml

    @cuml.setter
    def cuml(self, cuml: bool):
        """Use RAPIDS and cuml."""
        self._cuml = cuml

    @property
    def shard_size(self) -> int:
        """Maximum number of cells in dense arrays."""
        return self._shard_size

    @shard_size.setter
    def shard_size(self, shard_size: int):
        """Maximum number of cells in dense arrays."""
        self._shard_size = shard_size

    @property
    def seed(self) -> int:
        """Random seed for torch and numpy."""
        return self._seed

    @seed.setter
    def seed(self, seed: int):
        """Random seed for torch and numpy."""
        scvi.settings.seed = seed
        self._seed = seed

    @property
    def verbosity(self) -> int:
        """Verbosity level (default `logging.INFO`)."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, level: Union[str, int]):
        """
        Sets logging configuration for popV based on chosen level of verbosity.

        Parameters
        ----------
        level
            Sets "popV" logging level to `level`
        force_terminal
            Rich logging option, set to False if piping to file output.
        """
        self._verbosity = level
        popv_logger.setLevel(level)
        if len(popv_logger.handlers) == 0:
            console = Console(force_terminal=True)
            if console.is_jupyter is True:
                console.is_jupyter = False
            ch = RichHandler(
                level=level, show_path=False, console=console, show_time=False
            )
            formatter = logging.Formatter("%(message)s")
            ch.setFormatter(formatter)
            popv_logger.addHandler(ch)
        else:
            popv_logger.setLevel(level)

settings = Config()
