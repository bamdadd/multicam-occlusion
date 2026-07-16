"""Runnable demo: fuse complementary views at an assembly station.

    python examples/fusion_demo.py

Loads the assembly-station fixture (a multicam-sim scene manifest with
asymmetric per-camera visibility), routes the operator to camera 0 and the parts
to camera 1, correlates each operator action with the assembly change it caused,
verifies the result against the order, and prints the joint scene state, the
association metric, and the per-item order-verification status.
"""

from __future__ import annotations

from pathlib import Path

from multicam_occlusion.fusion import (
    CameraRoles,
    GroundTruthInteraction,
    SceneManifest,
    association_metric,
    fuse_scene,
    partition_by_visibility,
    verify_order,
)

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "assembly_station.json"
ORDER = ["part_a", "part_b"]

GROUND_TRUTH = [
    GroundTruthInteraction(actor_id="operator", item_id="part_a", action_time=0.5, change_time=0.6),
    GroundTruthInteraction(actor_id="operator", item_id="part_b", action_time=1.3, change_time=1.4),
]


def main() -> None:
    manifest = SceneManifest.from_json(FIXTURE)
    roles = CameraRoles(operator_camera=0, item_camera=1)

    partition = partition_by_visibility(manifest, roles)
    print(f"asymmetric routing:  actors={partition.actors}  items={partition.items}")

    state = fuse_scene(manifest, roles)
    print("\nfused interactions (who did what to which item, when):")
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
        f"\nassociation metric:  precision={m.precision:.2f}  recall={m.recall:.2f}  "
        f"f1={m.f1:.2f}  (TP={m.true_positives} FP={m.false_positives} FN={m.false_negatives})"
    )
    print(
        f"timing error:  action={m.mean_action_time_error:.3f}s  "
        f"change={m.mean_change_time_error:.3f}s  mean lag={m.mean_lag:.3f}s"
    )

    verified = verify_order(state, ORDER)
    print(f"\norder verification (order={ORDER}):  ok={verified.ok()}")
    for line in verified.lines:
        extra = f" (assembled {line.observed_item})" if line.observed_item else ""
        print(f"  {line.item_id}: {line.status.value}{extra}")


if __name__ == "__main__":
    main()
