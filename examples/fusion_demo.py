"""Runnable demo: fuse complementary views at a packing station.

    python examples/fusion_demo.py

Loads the packing-station fixture (a multicam-sim scene manifest with asymmetric
per-camera visibility), routes the worker to camera 0 and the items to camera 1,
correlates each worker action with the assembly change it caused, and prints the
joint scene state plus the association metric against ground truth.
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
)

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "packing_station.json"

GROUND_TRUTH = [
    GroundTruthInteraction(actor_id="worker", item_id="item_0", action_time=0.5, change_time=0.6),
    GroundTruthInteraction(actor_id="worker", item_id="item_1", action_time=1.3, change_time=1.4),
]


def main() -> None:
    manifest = SceneManifest.from_json(FIXTURE)
    roles = CameraRoles(human_camera=0, worktop_camera=1)

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


if __name__ == "__main__":
    main()
