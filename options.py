from copy import deepcopy
from dataclasses import dataclass
import functools
import json
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional


@dataclass
class Options:
    defines: Optional[dict[str, str]] = None
    extra_compile_options: Optional[List[str]] = None
    data_check: Optional[Literal["strict", "fuzzy"]] = None
    max_deviation: Optional[float] = None

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def __setitem__(self, item: str, value: Any) -> None:
        setattr(self, item, value)


DefaultOptions = Options(defines={}, extra_compile_options=[], data_check="strict")


def options_from_file(fp: Path) -> Options:
    ret = DefaultOptions
    with open(fp, "r") as src:
        j = json.load(src)
        for param in Options.__dataclass_fields__:
            if param in j:
                ret[param] = j[param]
    return ret


def override_options(parent: Options, child: Options) -> Options:
    ret = deepcopy(parent)
    for param in Options.__dataclass_fields__:
        if child[param] is not None:
            if type(child[param]) == list:
                ret[param] = parent[param] + child[param]
            elif type(child[param]) == dict:
                ret[param].update(child[param])
            else:
                ret[param] = child[param]
    return ret


def options_from_multiple_files(fps: Iterable[Path]) -> Options:
    # give it a list of filepaths and later filepaths will override prior ones
    read_options = map(options_from_file, fps)
    return functools.reduce(override_options, read_options, DefaultOptions)
