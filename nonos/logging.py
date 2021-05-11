import logging
import sys

from rich import print as rprint
from rich.logging import RichHandler


def print_warn(message):
    """
    adapted from idefix_cli (cmt robert)
    https://github.com/neutrinoceros/idefix_cli
    """
    rprint(f":bone: [bold red]Warning[/] {message}", file=sys.stderr)


def print_err(message):
    """
    adapted from idefix_cli (cmt robert)
    https://github.com/neutrinoceros/idefix_cli
    """
    rprint(f":bone: [bold white on red]Error[/] {message}", file=sys.stderr)


def logging_verbose(verbose):
    levels = ["WARNING", "INFO", "DEBUG"]
    level = levels[min(len(levels) - 1, verbose)]  # capped to number of levels

    FORMAT = "%(message)s"
    logging.basicConfig(
        level=level,
        force=True,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
