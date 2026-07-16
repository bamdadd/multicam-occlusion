"""Runnable demo: fuse complementary views at an assembly station.

    python examples/fusion_demo.py

Two halves of the fusion mode, each on the data it genuinely runs on:

1. **Order verification on REAL generated multicam-sim output.** Loads the
   committed generated fixture (``tests/fixtures/sim_assembly_station/``,
   regenerate with ``make fusion-scene``), routes the operator to camera 0 and
   the parts to camera 1 by asymmetric visibility, reconstructs the assembled
   contents from the item camera, and reconciles them against the generated
   order's BOM (fulfilled / missing / wrong / extra).

2. **Causal action↔change association on a CONTROLLED in-memory scene.** The
   sim preset frames an operator but does not yet emit reaches timestamped to
   placements, so the causal metric runs on a hand-authored controlled scene
   (``tests/controlled_assembly_scene.py``): it correlates each operator action
   with the assembly change it caused and scores precision/recall against known
   ground truth. Tracked upstream — see ``docs/fusion-design.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from multicam_occlusion.fusion import (
    CameraRoles,
    GroundTruthInteraction,
    SceneManifest,
    SceneOrder,
    association_metric,
    fuse_scene,
    partition_by_visibility,
    reconstruct_assembled,
    verify_order_from_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "sim_assembly_station"
ROLES = CameraRoles(operator_camera=0, item_camera=1)

# The controlled causal scene lives with the tests (not sim-generated).
sys.path.insert(0, str(REPO_ROOT / "tests"))
from controlled_assembly_scene import build as build_controlled_scene  # noqa: E402

CONTROLLED_ORDER = ["part_a", "part_b"]
GROUND_TRUTH = [
    GroundTruthInteraction(actor_id="operator", item_id="part_a", action_time=0.5, change_time=0.6),
    GroundTruthInteraction(actor_id="operator", item_id="part_b", action_time=1.3, change_time=1.4),
]


def order_verification_on_generated_data() -> None:
    """Half 1: order verification end-to-end on real multicam-sim output."""
    manifest = SceneManifest.from_json(SIM_FIXTURE / "manifest.json")
    order = SceneOrder.from_json(SIM_FIXTURE / "order.json")

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


def causal_association_on_controlled_scene() -> None:
    """Half 2: causal action<->change association on a controlled in-memory scene."""
    manifest = SceneManifest.model_validate(build_controlled_scene())

    print("\n== causal action<->change association on a CONTROLLED in-memory scene ==")
    state = fuse_scene(manifest, ROLES)
    print("fused interactions (who did what to which item, when):")
    for i in state.interactions:
        print(
            f"  {i.actor_id} {i.action_label!r} @ t={i.action_time:.1f}s "
            f"-> {i.item_id} {i.change!r} @ t={i.change_time:.1f}s "
            f"(lag {i.lag:.1f}s, conf {i.confidence:.2f})"
        )
    print("  refused (no causal partner):")
    for a in state.unassociated_actions:
        print(f"    action {a.label!r} @ t={a.time:.1f}s")
    for c in state.unassociated_changes:
        print(f"    change {c.item_id} @ t={c.time:.1f}s")

    m = association_metric(state.interactions, GROUND_TRUTH)
    print(
        f"association metric:  precision={m.precision:.2f}  recall={m.recall:.2f}  "
        f"f1={m.f1:.2f}  (TP={m.true_positives} FP={m.false_positives} FN={m.false_negatives})"
    )
    print(
        f"timing error:  action={m.mean_action_time_error:.3f}s  "
        f"change={m.mean_change_time_error:.3f}s  mean lag={m.mean_lag:.3f}s"
    )


def main() -> None:
    order_verification_on_generated_data()
    causal_association_on_controlled_scene()


if __name__ == "__main__":
    main()
