from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import queue
from typing import Any, Iterable, List, Tuple

"""
Handles collecting all tasks to run and dispatches them.
"""


class Runner:

    def __init__(self) -> None:
        self.pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
        self.q = queue.Queue()

    def run_tasks(
        self, tasks: List[Tuple[Callable, List[Any]]], parallel=True
    ) -> Iterable[Any]:
        if parallel:
            for task, args in tasks:
                self.q.put(self.pool.submit(task, *args))
            for task in tasks:
                yield self.q.get().result()
        else:
            for task, args in tasks:
                yield task(*args)
