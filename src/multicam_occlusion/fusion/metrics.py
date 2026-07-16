"""The association metric: is the fused (action <-> change) pairing correct?

This generalizes "did we correctly link the box's contents to the open order?"
to "did we correctly link each worker action to the assembly change it caused?"
The score is computed over (action, item-change) *pairs*, not over frames or
points, because the pair is the unit the mode is supposed to recover.

A predicted :class:`FusedInteraction` is a **true positive** when a ground-truth
interaction exists with the same ``(actor_id, item_id)`` and an action time
within ``time_tol`` seconds. From that matching:

* **precision** = TP / predicted  — of the links we asserted, how many are real
  (a spurious link to a distractor action costs precision).
* **recall** = TP / ground-truth  — of the real links, how many we recovered.
* **timing error** = mean absolute error, over matched pairs, of the action time
  and of the change time versus ground truth (plus the mean fused lag).

Matching is greedy one-to-one, so a single prediction cannot claim two
ground-truth pairs (or vice versa).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .observations import FusedInteraction, GroundTruthInteraction


class AssociationMetrics(BaseModel):
    """Precision/recall/F1 over (action, change) pairs, plus timing error."""

    model_config = ConfigDict(frozen=True)

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_action_time_error: float
    mean_change_time_error: float
    mean_lag: float


def association_metric(
    predicted: list[FusedInteraction],
    ground_truth: list[GroundTruthInteraction],
    time_tol: float = 0.2,
) -> AssociationMetrics:
    """Score predicted interactions against ground-truth (action, change) pairs.

    Args:
        predicted: the estimator's :class:`FusedInteraction`s.
        ground_truth: the known pairings (by construction of the scene).
        time_tol: max ``|action_time|`` difference (seconds) for a match.

    Returns:
        :class:`AssociationMetrics`. Empty-input edge cases are defined:
        precision is ``1.0`` when nothing was predicted (nothing wrong asserted),
        recall is ``1.0`` when there is nothing to recover.
    """
    matched_gt: set[int] = set()
    tp = 0
    action_errs: list[float] = []
    change_errs: list[float] = []
    lags: list[float] = []

    for pred in predicted:
        best_gi: int | None = None
        best_err: float | None = None
        for gi, gt in enumerate(ground_truth):
            if gi in matched_gt:
                continue
            if pred.actor_id != gt.actor_id or pred.item_id != gt.item_id:
                continue
            err = abs(pred.action_time - gt.action_time)
            if err > time_tol:
                continue
            if best_err is None or err < best_err:
                best_err, best_gi = err, gi
        if best_gi is not None:
            matched_gt.add(best_gi)
            gt = ground_truth[best_gi]
            tp += 1
            action_errs.append(abs(pred.action_time - gt.action_time))
            change_errs.append(abs(pred.change_time - gt.change_time))
            lags.append(pred.lag)

    fp = len(predicted) - tp
    fn = len(ground_truth) - tp
    precision = tp / len(predicted) if predicted else 1.0
    recall = tp / len(ground_truth) if ground_truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return AssociationMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_action_time_error=_mean(action_errs),
        mean_change_time_error=_mean(change_errs),
        mean_lag=_mean(lags),
    )
