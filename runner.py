from abc import ABC, abstractmethod
from collections.abc import Callable
import threading
from typing import Any, Iterable

"""
Handles collecting all tasks to run and dispatches them.
"""


class Runner:
    def run_tasks(self, tasks: Iterable[Callable]) -> Iterable[Any]:
        for task in tasks:
            yield task()
