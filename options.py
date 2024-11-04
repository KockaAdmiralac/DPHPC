from copy import deepcopy
from dataclasses import dataclass, field
import functools
import json
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional

DefinesConstraints = list[dict[str, int | str]]


@dataclass
class Options:
    defines: dict[str, str] = field(default_factory=lambda: {})
    extra_compile_options: List[str] = field(default_factory=lambda: [])
    data_check: Literal["strict", "fuzzy"] = "strict"
    max_deviation: Optional[float] = None
    defines_constraints: DefinesConstraints = field(default_factory=lambda: [])

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def __setitem__(self, item: str, value: Any) -> None:
        setattr(self, item, value)


DefaultOptions = Options()


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
