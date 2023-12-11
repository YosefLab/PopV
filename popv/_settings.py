import logging
from pathlib import Path
from typing import Union

import torch
from rich.console import Console
from rich.logging import RichHandler

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

    To set the number of threads to be used

    >>> scvi.settings.num_threads = 2

    """

    def __init__(
        self,
        verbosity: int = logging.WARNING,
        seed: int = 0,
        logging_dir: str = "./popv_log/",
    ):
        """Set up Config manager for PopV."""
        self.seed = seed
        self._num_threads = None
        self.logging_dir = logging_dir
        self.verbosity = verbosity

    @property
    def logging_dir(self) -> Path:
        """Directory for training logs (default `'./popv_log/'`)."""
        return self._logging_dir

    @logging_dir.setter
    def logging_dir(self, logging_dir: Union[str, Path]):
        self._logging_dir = Path(logging_dir).resolve()

    @property
    def num_threads(self) -> None:
        """Number of threads in PyTorch."""
        return self._num_threads

    @num_threads.setter
    def num_threads(self, num: int):
        """Number of threads in PyTorch."""
        self._num_threads = num
        torch.set_num_threads(num)

    @property
    def seed(self) -> int:
        """Random seed for torch and numpy."""
        return self._seed

    @seed.setter
    def seed(self, seed: int):
        """Random seed for torch and numpy."""
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self._seed = seed

    @property
    def verbosity(self) -> int:
        """Verbosity level (default `logging.INFO`)."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, level: Union[str, int]):
        """
        Sets logging configuration for popv based on chosen level of verbosity.

        If "scvi" logger has no StreamHandler, add one.
        Else, set its level to `level`.

        Parameters
        ----------
        level
            Sets "popv" logging level to `level`
        force_terminal
            Rich logging option, set to False if piping to file output.
        """
        self._verbosity = level
        popv_logger.setLevel(level)
        if len(popv_logger.handlers) == 0:
            console = Console(force_terminal=True)
            if console.is_jupyter is True:
                console.is_jupyter = False
            ch = RichHandler(level=level, show_path=False, console=console, show_time=False)
            formatter = logging.Formatter("%(message)s")
            ch.setFormatter(formatter)
            popv_logger.addHandler(ch)
        else:
            popv_logger.setLevel(level)
