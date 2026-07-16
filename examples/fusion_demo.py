"""Runnable demo: fuse complementary views at an assembly station.

    python examples/fusion_demo.py

Both metrics of the fusion mode, end-to-end on REAL generated multicam-sim output
(the committed fixture ``tests/fixtures/sim_assembly_station/``, regenerate with
``make fusion-scene``):

1. **Order verification.** Route the operator to camera 0 and the parts to camera
   1 by asymmetric visibility, reconstruct the assembled contents from the item
   camera, reconcile against the generated order's BOM (fulfilled / missing /
   wrong / extra).

2. **Causal action↔change association.** Consume the placement-synced operator
   ``actions[]`` from the generated ``order.json``, pair each against the item's
   assembly change detected from the manifest, and score precision/recall/timing
   against the producer's declared ground truth.

The estimator's distractor-refusal behaviour (a reach that assembles nothing, a
change outside the lag window) is a separate unit test on a controlled scene —
this clean generated scene contains no distractors.
"""

from __future__ import annotations

from pathlib import Path

from multicam_occlusion.fusion import (
    CameraRoles,
    SceneManifest,
    SceneOrder,
    association_metric,
    fuse_order_actions,
    ground_truth_from_order,
    partition_by_visibility,
    reconstruct_assembled,
    verify_order_from_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "sim_assembly_station"
ROLES = CameraRoles(operator_camera=0, item_camera=1)


def order_verification(manifest: SceneManifest, order: SceneOrder) -> None:
    """Order verification end-to-end on real multicam-sim output."""
    print("== order verification on REAL generated multicam-sim output ==")
    partition = partition_by_visibility(manifest, ROLES)
    print(f"asymmetric routing:  actors={partition.actors}  items={partition.items}")

    assembled = reconstruct_assembled(manifest, ROLES)
    print(f"assembled (from item camera):  {assembled}")
    print(f"order BOM (expected):          {order.expected_counts()}")

    verified = verify_order_from_manifest(manifest, order, ROLES)
    print(f"order verification:  ok={verified.ok()}")
    for line in verified.lines:
        print(
            f"  {line.item_id}: {line.status.value}  "
            f"(expected {line.expected}, assembled {line.assembled})"
        )


def causal_association(manifest: SceneManifest, order: SceneOrder) -> None:
    """Causal action<->change association on real multicam-sim output."""
    print("\n== causal action<->change association on REAL generated output ==")
    print(f"operator actions from order.json:  {[a.item_id for a in order.actions]}")

    state = fuse_order_actions(manifest, order, ROLES)
    print("fused interactions (who did what to which item, when):")
    for i in state.interactions:
        print(
            f"  {i.actor_id} {i.action_label!r} @ t={i.action_time:.3f}s "
            f"-> {i.item_id} {i.change!r} @ t={i.change_time:.3f}s "
            f"(lag {i.lag:.3f}s, conf {i.confidence:.2f})"
        )

    m = association_metric(state.interactions, ground_truth_from_order(order, manifest))
    print(
        f"association metric:  precision={m.precision:.2f}  recall={m.recall:.2f}  "
        f"f1={m.f1:.2f}  (TP={m.true_positives} FP={m.false_positives} FN={m.false_negatives})"
    )
    print(
        f"timing error:  action={m.mean_action_time_error:.3f}s  "
        f"change={m.mean_change_time_error:.3f}s  mean lag={m.mean_lag:.3f}s"
    )


def main() -> None:
    manifest = SceneManifest.from_json(SIM_FIXTURE / "manifest.json")
    order = SceneOrder.from_json(SIM_FIXTURE / "order.json")
    order_verification(manifest, order)
    causal_association(manifest, order)


if __name__ == "__main__":
    main()
