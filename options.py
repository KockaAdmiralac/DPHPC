from copy import deepcopy
from dataclasses import dataclass, field
import functools
import json
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional
from structures import *


DefaultOptions = Options()


def options_from_file(fp: Path) -> Options:
    ret = deepcopy(DefaultOptions)
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
                ret[param] = child[param]
            elif type(child[param]) == dict:
                ret[param].update(child[param])
            else:
                ret[param] = child[param]
    return ret


def options_from_multiple_files(fps: Iterable[Path]) -> Options:
    # give it a list of filepaths and later filepaths will override prior ones
    read_options = map(options_from_file, fps)
    return functools.reduce(override_options, read_options, DefaultOptions)
