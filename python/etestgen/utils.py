from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
import collections
import numpy as np
import seutil as su
from scipy import stats
from collections import defaultdict
from etestgen.macros import Macros

logger = su.log.get_logger(__name__)


SUMMARIES_FUNCS: Dict[str, Callable[[Union[list, np.ndarray]], Union[int, float]]] = {
    "AVG": lambda l: np.mean(l).item() if len(l) > 0 else np.NaN,
    "SUM": lambda l: np.sum(l).item() if len(l) > 0 else np.NaN,
    "MAX": lambda l: np.max(l).item() if len(l) > 0 else np.NaN,
    "MIN": lambda l: np.min(l).item() if len(l) > 0 else np.NaN,
    "MEDIAN": lambda l: (
        np.median(l).item() if len(l) > 0 and np.NaN not in l else np.NaN
    ),
    "STDEV": lambda l: np.std(l).item() if len(l) > 0 else np.NaN,
    "MODE": lambda l: stats.mode(l).mode[0].item() if len(l) > 0 else np.NaN,
    "CNT": lambda l: len(l),
}


SUMMARIES_PRESERVE_INT: Dict[str, bool] = {
    "AVG": False,
    "SUM": True,
    "MAX": True,
    "MIN": True,
    "MEDIAN": False,
    "STDEV": False,
    "MODE": True,
    "CNT": True,
}


def summarize_metrics(
    metrics: Dict[str, Union[float, List[float]]],
) -> Dict[str, float]:
    metrics_summary = {}
    for k, v in metrics.items():
        if "topk" in k:
            continue
        if isinstance(v, list):
            metrics_summary[k] = float(np.mean([float(x) for x in v]))
        else:
            metrics_summary[k] = float(v)
    return metrics_summary


def update_metrics(
    metrics: Dict[str, Union[float, List[float]]],
    metrics_new: Dict[str, Union[float, List[float]]],
):
    for k, v in metrics_new.items():
        if k not in metrics:
            metrics[k] = v
        else:
            if isinstance(v, list) and isinstance(metrics[k], list):
                metrics[k] += v
            else:
                raise ValueError(
                    f"Can only update list metrics into list metrics, but {k} is {type(v)} vs. {type(metrics[k])}"
                )


def aggregate_metrics(
    metrics_list: List[Dict[str, Union[float, List[float]]]],
) -> Dict[str, List[float]]:
    metrics_agg = {}
    for metrics in metrics_list:
        for k, v in metrics.items():
            if k not in metrics_agg:
                metrics_agg[k] = [v]
            else:
                if isinstance(v, list):
                    metrics_agg[k] += v
                else:
                    metrics_agg[k].append(v)
    return metrics_agg


def summarize_metrics_rich(
    metrics: Dict[str, Union[float, List[float]]],
) -> Dict[str, float]:
    metrics_summary = {}
    for k, v in metrics.items():
        if isinstance(v, list):
            for s, f in SUMMARIES_FUNCS.items():
                metrics_summary[f"{k}-{s}"] = f(v)
        else:
            metrics_summary[k] = v
    return metrics_summary


TElem = TypeVar("TElem")
TValue = TypeVar("TValue")


def compute_similarity_metrics(
    gold: List[str],
    topk: List[List[str]],
    k_values: List[int] = None,
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    if k_values is None:
        k_values = [1]

    metrics = {}

    # compute all required metrics
    if len(topk) == 0:
        # dummy empty prediction, to simplify corner case handling
        topk = [[]]
    metrics_all = collections.defaultdict(list)

    # HACK put import here so it doesn't break the whole program when the import fails
    from etestgen.eval.metrics import bleu, code_bleu, edit_sim, rouge_l

    for pred in topk:
        metrics_all["xmatch"].append(100 if gold == pred else 0)
        metrics_all["bleu"].append(bleu(gold, pred))
        metrics_all["code-bleu"].append(code_bleu(gold, pred))
        metrics_all["edit-sim"].append(edit_sim(gold, pred))
        for k, v in rouge_l(gold, pred).items():
            metrics_all[f"rouge-{k}"].append(v)

    # xmatch and xmatch-topk
    metrics["xmatch"] = metrics_all["xmatch"][0]
    for k in k_values:
        metrics[f"xmatch-top{k}"] = max(metrics_all["xmatch"][:k])

    # similarity metrics
    for m in [
        "bleu",
        "code-bleu",
        "edit-sim",
        "rouge-p",
        "rouge-r",
        "rouge-f",
        "xmatch",
    ]:
        # top1
        metrics[m] = metrics_all[m][0]
        # max, min, avg
        metrics[f"{m}-max"] = max(metrics_all[m])
        metrics[f"{m}-min"] = min(metrics_all[m])
        metrics[f"{m}-avg"] = sum(metrics_all[m]) / len(metrics_all[m])
        # weighted avg
        if weights is not None:
            metrics[f"{m}-wavg"] = sum(
                [x * w for x, w in zip(metrics_all[m], weights)]
            ) / sum(weights)

    return metrics


def summarize_topk_metrics(topk_metrics: Dict[str, List[float]]) -> dict:
    """
    Summarize the topk metrics: max, min, avg.
    """
    metrics = {}
    for m, v in topk_metrics.items():
        metrics[f"{m}-max"] = max(topk_metrics[m])
        metrics[f"{m}-min"] = min(topk_metrics[m])
        metrics[f"{m}-avg"] = sum(topk_metrics[m]) / len(topk_metrics[m])
        metrics[f"{m}-topk"] = topk_metrics[m]
    #
    return metrics


def load_test_projects():
    projects = su.io.load(
        Macros.results_dir / "repos" / "mut2e-repos.json", su.io.Fmt.json
    )
    # test projects have 32 projects, but only 30 projects have throw statements in public methods
    # so we changed to load projects from results/tool-results/rq2-public-target-classes.json
    # test_projects = su.io.load(
    #     Macros.results_dir / "repos" / "test-projects.json", su.io.Fmt.json
    # )
    test_projects = su.io.load(
        Macros.results_dir / "tool-results" / "rq2-public-target-classes.json"
    ).keys()
    projects = [p for p in projects if p["full_name"] in test_projects]
    return projects


def configure_tests(proj: str):
    # remove maven.config
    if proj == "arquillian_arquillian-core":
        su.bash.run(f"rm {Macros.downloads_dir}/{proj}/.mvn/maven.config")


class Trie(Generic[TElem, TValue]):
    """
    Trie structure to store mapping from iterables to values.
    """

    def __init__(
        self,
        empty_elem: TElem = "",
        join_op: Optional[Callable[[Iterable[TElem]], Any]] = lambda x: "".join(x),
    ):
        if join_op is None:
            join_op = lambda x: x

        self.empty_elem = empty_elem
        self.data = {}
        self.join_op = join_op

    def get(
        self, key: Iterable[TElem], default: Optional[TValue] = None
    ) -> Optional[TValue]:
        cur = self.data
        for c in key:
            if c not in cur:
                return default
            cur = cur[c]
        return cur.get(self.empty_elem, default)

    def set(
        self, key: Iterable[TElem], value: TValue = None, exist_ok: bool = True
    ) -> Optional[TValue]:
        cur = self.data
        for c in key:
            if c not in cur:
                cur[c] = {}
            cur = cur[c]
        if not exist_ok and self.empty_elem in cur:
            raise KeyError(f"Key {key} already exists in the trie")
        else:
            old_value = cur.get(self.empty_elem, None)
            cur[self.empty_elem] = value
            return old_value

    def update_value(
        self, key: Iterable[TElem], update_op: Callable[[Optional[TValue]], TValue]
    ) -> Optional[TValue]:
        cur = self.data
        for c in key:
            cur.setdefault(c, {})
            cur = cur[c]
        old_value = cur.get(self.empty_elem, None)
        cur[self.empty_elem] = update_op(old_value)
        return old_value

    def remove(self, key: Iterable[TElem]) -> TValue:
        cur = self.data
        for c in key:
            if c not in cur:
                raise KeyError(f"Key {key} does not exist in the trie")
            cur = cur[c]
        if self.empty_elem not in cur:
            raise KeyError(f"Key {key} does not exist in the trie")
        value = cur[self.empty_elem]
        del cur[self.empty_elem]
        return value

    def has_key(self, key: Iterable[TElem]) -> bool:
        cur = self.data
        for c in key:
            if c not in cur:
                return False
            cur = cur[c]
        return self.empty_elem in cur

    def has_prefix(self, prefix: Iterable[TElem]) -> bool:
        cur = self.data
        for c in prefix:
            if c not in cur:
                return False
            cur = cur[c]
        return True

    def get_subtrie(self, prefix: Iterable[TElem]) -> "Trie[TElem, TValue]":
        """
        Get a subtrie with the given prefix, could be empty.
        """
        cur = self.data
        for c in prefix:
            if c not in cur:
                return Trie(self.empty_elem, self.join_op)
            cur = cur[c]
        trie = Trie(self.empty_elem, self.join_op)
        trie.data = cur
        return trie

    def get_root_value(self) -> Optional[TValue]:
        return self.data.get(self.empty_elem, None)

    def values_prefix_of(
        self, key: Iterable[TElem], include_initial_empty: bool = False
    ) -> Iterable[TValue]:
        """
        Get all values whose keys are prefixes of the given key.
        """
        cur = self.data

        if include_initial_empty and self.empty_elem in cur:
            yield cur[self.empty_elem]

        for c in key:
            if c in cur:
                cur = cur[c]
                if self.empty_elem in cur:
                    yield cur[self.empty_elem]
            else:
                break

    def items_prefix_of(
        self, key: Iterable[TElem], include_initial_empty: bool = False
    ) -> Iterable[Tuple[Iterable[TElem], TValue]]:
        """
        Get all values whose keys are prefixes of the given key.
        """
        cur = self.data

        if include_initial_empty and self.empty_elem in cur:
            yield self.empty_elem, cur[self.empty_elem]

        prefix = []
        for c in key:
            prefix.append(c)
            if c in cur:
                cur = cur[c]
                if self.empty_elem in cur:
                    yield self.join_op(prefix), cur[self.empty_elem]
            else:
                break

    def values_all_in(
        self, keys: Set[TElem], include_initial_empty: bool = False
    ) -> Iterable[TValue]:
        """
        Get all values whose keys are all in the given set.
        """
        if include_initial_empty and self.empty_elem in self.data:
            yield self.data[self.empty_elem]

        queue = [self.data]
        while len(queue) > 0:
            cur = queue.pop(0)
            if cur is not self.data or include_initial_empty:
                if self.empty_elem in cur:
                    yield cur[self.empty_elem]

            for c in keys:
                if c in cur:
                    queue.append(cur[c])

    def subtries(self) -> Iterable[Tuple[TElem, "Trie[TElem, TValue]"]]:
        for k, v in self.data.items():
            if k != self.empty_elem:
                subtrie = Trie(self.empty_elem, self.join_op)
                subtrie.data = v
                yield k, subtrie

    def items(self) -> Iterable[Tuple[Iterable[TElem], TValue]]:
        queue = [([], self.data)]
        while len(queue) > 0:
            prefix, cur = queue.pop()
            for k, v in cur.items():
                if k == self.empty_elem:
                    yield (self.join_op(prefix), v)
                else:
                    queue.append((prefix + [k], v))

    def keys(self) -> Iterable[Iterable[TElem]]:
        queue = [([], self.data)]
        while len(queue) > 0:
            prefix, cur = queue.pop()
            for k, v in cur.items():
                if k == self.empty_elem:
                    yield self.join_op(prefix)
                else:
                    queue.append((prefix + [k], v))

    def values(self) -> Iterable[TValue]:
        queue = [([], self.data)]
        while len(queue) > 0:
            prefix, cur = queue.pop()
            for k, v in cur.items():
                if k == self.empty_elem:
                    yield v
                else:
                    queue.append((prefix + [k], v))

    def __len__(self) -> int:
        return sum(1 for _ in self.values())

    def __str__(self):
        return f"Trie(empty_value={self.empty_elem}, data={self.data})"

    def __repr__(self) -> str:
        return self.__str__()

    def overlaps_with(
        self, other: "Trie", self_has_value: bool = False, other_has_value: bool = False
    ) -> Iterable[Tuple["Trie", "Trie"]]:
        """
        Find all subtries that overlap with the other trie.
        Does not include the empty subtrie.
        :param other: the other trie to compare with.
        :param self_has_value: whether to only output overlaps where the self subtrie has a value.
        :param other_has_value: whether to only output overlaps where the other subtrie has a value.
        """
        queue = [(self.data, other.data, [])]
        skipped_root = False
        while len(queue) > 0:
            self_node, other_node, prefix = queue.pop(0)
            if not skipped_root:
                skipped_root = True
            else:
                # check if we should output this overlap
                if ((not self_has_value) or (self.empty_elem in self_node)) and (
                    (not other_has_value) or (other.empty_elem in other_node)
                ):
                    self_subtrie = Trie(self.empty_elem, self.join_op)
                    self_subtrie.data = self_node
                    other_subtrie = Trie(other.empty_elem, other.join_op)
                    other_subtrie.data = other_node
                    yield self_subtrie, other_subtrie

            # find all overlapping children
            for k, v in self_node.items():
                if k == self.empty_elem:
                    continue
                if k in other_node:
                    queue.append((v, other_node[k], prefix + [k]))

def aggregate_predictions_by_id(predictions: List, dataset: List):
    """
    Aggregate the predictions w.r.t the prompt by id and return the dict[List]
    """
    id_2_topk = defaultdict(list)
    for pred, dt in zip(predictions, dataset):
        data_id, pred_str = pred
        id_2_topk[data_id].append((dt["instruction"], pred_str))
    return id_2_topk
