import sys

from rich import print as rprint


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
