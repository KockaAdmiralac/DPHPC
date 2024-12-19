from typing import Any, Iterable, List, Tuple

from numpy import ndarray
import numpy

from structures import ProcessedResult, SingleBenchmark
import scipy.stats

ResultTimings = dict[str, List[float]]
PreprocessedResultPair = Tuple[SingleBenchmark, ResultTimings]
PreprocessedResults = List[PreprocessedResultPair]


def get_ci(
    data: ndarray,
    statistic: Any,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
):
    for_scipy = (data,)
    bs = scipy.stats.bootstrap(
        for_scipy, statistic, n_resamples=n_resamples, confidence_level=confidence_level
    )
    return bs.confidence_interval


def get_ci_from_results(
    data: Iterable[ProcessedResult],
    timing_name: str,
    statistic: str,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
):
    np_ver = numpy.fromiter((x.timings[timing_name] for x in data), float)
    return get_ci(np_ver, statistic, n_resamples, confidence_level)


def keep_valid_runs(results: Iterable[ProcessedResult]) -> Iterable[ProcessedResult]:
    for res in results:
        if bool(res.timings):
        #if res.data_checked and res.data_valid:
            yield res


def group_runs(results: Iterable[ProcessedResult]) -> PreprocessedResults:
    existing_results: PreprocessedResults = []
    for res in results:
        curr_tuple = next(
            (t for t in existing_results if t[0] == res.referenced_run), None
        )
        if curr_tuple is None:
            curr_tuple = (res.referenced_run, {})
            existing_results.append(
                curr_tuple
            )  # note curr_tuple will keep the reference to this
        curr_ret = curr_tuple[1]
        for t, val in res.timings.items():
            if t not in curr_ret:
                curr_ret[t] = []
            curr_ret[t].append(val)
    return existing_results


def preprocess_results(
    results: Iterable[ProcessedResult],
) -> PreprocessedResults:
    return group_runs(keep_valid_runs(results))
