"""Verify graph invariants and render native CT and repair audit panels."""

from argparse import ArgumentParser
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np

from native_ct import components, ct_route
from run_topology import BASELINE, EDGE, OUT, load_ct_training, load_graph, sha, write_json
from section_graph import SectionGraph


def check_graph(graph):
    nodes = {n["id"]: n for n in graph["nodes"]}
    expected = components(nodes, [e["nodes"] for e in graph["edges"]])
    assert expected == graph["components"], graph["source"]
    assert len({tuple(sorted(e["nodes"])) for e in graph["edges"]}) == len(graph["edges"])
    for e in graph["edges"]:
        a, b = (nodes[i] for i in e["nodes"])
        if e["provenance"] in ("sampled", "inferred"):
            assert a["name"] == b["name"] and a["surface_component"] == b["surface_component"], e
        assert SectionGraph.vascular_class(a["name"]) == SectionGraph.vascular_class(b["name"]), e
        if e["provenance"] == "sampled":
            assert b["station_mm"] == a["station_mm"] + 1, e
        if e["provenance"] == "inferred":
            assert e["evidence"]["supported_probes"] >= .8 * e["evidence"]["probes"], e
            assert e["evidence"]["max_blank_run_mm"] <= 2., e
        if e["provenance"] == "inferred_junction":
            assert e["evidence"]["local_vertex_pairs_le_0p8mm"] >= 3, e
    return {"nodes": len(nodes), "edges": len(graph["edges"]), "components": len(expected)}


def verify(out, base, rebuild=None):
    native = load_graph(out / "medseg_native_graph.json")
    labeled, edges = ct_route(native)
    ct = load_ct_training(out / "medseg_ct_route.npz")
    assert len(labeled) == 31 and len(edges) == 30
    assert ct["native_layer"].tolist() == list(range(31))
    np.testing.assert_allclose(ct["native_ras_mm"][:, 2], np.arange(31) * 3.)
    np.testing.assert_array_equal(ct["label_id"], [39] * 13 + [23] * 18)
    np.testing.assert_array_equal(ct["source_role_label_ids"][:, 0], ct["label_id"])
    assert ct["shared_voxel_faces"][12] == 233 and ct["shared_voxel_faces"].min() >= 1
    assert np.isfinite(ct["shape_centers_xz_mm"][:, :, 0]).all()
    assert (np.diff(ct["shape_stations_mm"], axis=1) > 0.).all()
    # CT projections never borrow a source label after its native coverage ends.
    assert np.array_equal(np.isfinite(ct["shape_centers_xz_mm"]).all(axis=-1),
                          np.broadcast_to(ct["source_role_available"], (7, 31, 6)))
    training = load_graph(out / "ct_reference_topology.json.gz")
    graphs = training["sources"]
    assert not any("Blender" in s for s in graphs)
    checks = {name: check_graph(graph) for name, graph in graphs.items()}
    held = load_graph(out / "blender_holdout_topology.json.gz")["source"]
    checks[held["source"]] = check_graph(held)
    assert not any(n["provenance"] == "interpolated" for n in held["nodes"])
    assert sum(e["provenance"] == "source_junction" for e in held["edges"]) == 4
    with np.load(out / "ct_geometry.npz", allow_pickle=False) as a:
        lo, hi = a["ct_bounds"]
        assert hi - lo == 16 and np.all(a["status"][3, lo:hi + 1, 0] == 1)
        assert np.all(a["links"][3, lo:hi, 0])
        assert np.count_nonzero(a["edge_status"][3, lo:hi, 0] == EDGE["source_junction"]) == 1
        assert np.array_equal(a["links"], a["edge_status"] > 0)
        np.testing.assert_array_equal(a["links"].shape, (4, 160, 6))
    with np.load(out / "blender_holdout_geometry_links.npz", allow_pickle=False) as a:
        np.testing.assert_array_equal(a["neutral_grid"], np.arange(-510., -1281., -5.))
        np.testing.assert_array_equal(a["links"].shape, (154, 6))
        assert np.array_equal(a["links"], a["edge_status"] > 0)
        legacy = base.parents[1] / "geometry.npz"
        if legacy.is_file():
            with np.load(legacy, allow_pickle=False) as z:
                present = np.isfinite(a["centers"]).all(axis=-1)
                legacy_present = np.isfinite(z["centers"][3]).all(axis=-1)
                both = present & legacy_present
                diff = np.linalg.norm(a["centers"] - z["centers"][3], axis=-1)
                checks["blender_projection_legacy_comparison"] = {
                    "matched_finite_station_role_pairs": int(both.sum()),
                    "coverage_status_disagreements": int(np.sum(present != legacy_present)),
                    "discrepant_centers_gt_1e-6_mm": int(np.sum(diff[both] > 1.e-6)),
                    "max_absolute_discrepancy_mm": float(diff[both].max())}
    result = {"status": "passed", "checks": checks,
              "input_hashes": {p.name: sha(p) for p in (
                  out / "ct_geometry.npz", out / "ct_reference_topology.json.gz",
                  out / "medseg_ct_route.npz", out / "medseg_native_graph.json",
                  out / "blender_holdout_geometry_links.npz", out / "blender_holdout_topology.json.gz")}}
    if rebuild is not None:
        artifacts = [*result["input_hashes"], "training_topology_audit.json",
                     "blender_holdout_topology_audit.json"]
        repeated = {name: sha(out / name) == sha(rebuild / name) for name in artifacts}
        if not all(repeated.values()):
            raise ValueError(f"Rebuilt graph outputs differ: {[k for k, v in repeated.items() if not v]}")
        result["offline_rebuild_byte_identical"] = repeated
    write_json(out / "verification.json", result)
    return ct, training


def render(out, ct, training):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    from matplotlib import font_manager
    regular = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if regular.is_file():
        font_manager.fontManager.addfont(regular)
        family = font_manager.FontProperties(fname=regular).get_name()
    else:
        family = "Noto Sans CJK SC"
    plt.rcParams.update({"font.family": family, "font.size": 11,
                         "axes.unicode_minus": False})
    palette = {23: "#285f90", 39: "#39806a"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.7),
                                   gridspec_kw={"width_ratios": [1.2, 1]})
    fig.patch.set_facecolor("white")
    z = ct["native_ras_mm"][:, 2]
    y = ct["native_ras_mm"][:, 1]
    label = ct["label_id"]
    for i, (a, b) in enumerate(pairwise(label)):
        ax1.plot(z[i:i + 2], y[i:i + 2], color=palette[a] if a == b else "#bc771a",
                 lw=1.85, zorder=1, linestyle="-" if a == b else "--")
    for lid, name in ((23, "右股总静脉 · 23"), (39, "右股静脉 · 39")):
        mask = label == lid
        ax1.scatter(z[mask], y[mask], s=32, label=name, color=palette[lid], zorder=2)
    ax1.axvline(37.5, color="#bc771a", alpha=.55, linestyle=":", lw=1)
    ax1.set(xlabel="原生 CT 轴向位置（mm）", ylabel="原生 RAS 前后坐标（mm）",
            title="31 层均有目标标注")
    ax1.xaxis.set_major_locator(MultipleLocator(15))
    ax1.legend(loc="upper right", frameon=False)

    colors = [palette[i] for i in label[:-1]]
    colors[12] = "#bc771a"
    ax2.bar(z[:-1] + 1.5, ct["shared_voxel_faces"], color=colors, width=2.3)
    ax2.set(xlabel="相邻层交界轴向位置（mm）", ylabel="共享体素面数",
            title="30 对相邻层均直接连通")
    ax2.xaxis.set_major_locator(MultipleLocator(15))
    ax2.annotate("标签交界：233 个面", xy=(37.5, 233), xytext=(5, 448),
                 arrowprops={"arrowstyle": "-", "color": "#bc771a"}, color="#76511d")
    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#e4e8ec", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("MedSeg 右侧股总静脉与股静脉", fontsize=18, x=.08, ha="left")
    fig.text(.08, .02, "一例原生 CT · 轴向间距 3 mm · 层中心跨度 90 mm · 1 个连通分量 · 0 个插值节点",
             fontsize=10, color="#405567")
    fig.subplots_adjust(left=.09, right=.978, bottom=.16, top=.78, wspace=.30)
    fig.savefig(out / "medseg_ct_route.png", dpi=155)
    plt.close(fig)

    open3d = training["sources"]["Open3DModel right"]
    joins = [a for a in open3d["gaps_and_join_candidates"] if a["status"] == "inferred_same_identity_bridge"]
    fig, ax = plt.subplots(figsize=(8, 4.15))
    for event in joins:
        first, last = event["from_mm"], event["to_mm"]
        rows = sorted((n for n in open3d["nodes"] if n["name"] == event["name"]
                       and first - 6 <= n["station_mm"] <= last + 6), key=lambda n: n["station_mm"])
        a, b = (open3d["nodes"][i] for i in event["nodes"])
        for inferred, desc, styling in (
                (False, "原图切面样本", {"c": "#285f90", "marker": "o"}),
                (True, "推断点", {"c": "#bc771a", "marker": "x"})):
            selected = [n for n in rows if (n["provenance"] == "interpolated") == inferred]
            ax.scatter([n["station_mm"] for n in selected], [n["xyz_mm"][2] for n in selected],
                       label=desc, **styling)
        ax.plot((first, last), (a["xyz_mm"][2], b["xyz_mm"][2]), c="#bc771a", linestyle="--", lw=1)
        ax.axvspan(first, last, color="#bc771a", alpha=.09)
        ax.text(.02, .02, f"区间内每 0.5 mm 复切，{event['supported_probes']}/{event['probes']} 个源曲面探针有闭环",
                transform=ax.transAxes, fontsize=10, color="#405567")
    ax.set(xlabel="距 SMPL-X 轴向原点（mm）", ylabel="中立位前后坐标（mm）",
           title="Open3DModel 右大隐静脉断点修复")
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e8ec", lw=.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "open3d_vein_gap.png", dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--rebuild", type=Path)
    opts = parser.parse_args()
    render(opts.out, *verify(opts.out, opts.baseline, opts.rebuild))
