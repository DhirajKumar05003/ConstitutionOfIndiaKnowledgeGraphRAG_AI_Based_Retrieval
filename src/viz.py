"""
viz.py
=======
Renders the knowledge graph (or a highlighted view of it) as an interactive
pyvis/vis-network HTML fragment, suitable for embedding in Streamlit via
`st.components.v1.html`.
"""

from __future__ import annotations

from pyvis.network import Network

from src.graph_store import ConstitutionGraphStore

GROUP_COLORS = {
    "part": "#ffd166",
    "article": "#06d6a0",
    "clause": "#118ab2",
    "subclause": "#ef476f",
}
GROUP_SIZES = {"part": 30, "article": 18, "clause": 12, "subclause": 8}
# Fixed depth per node type -> feeds vis.js's hierarchical layout so the
# graph always reads top-to-bottom as Part -> Article -> Clause -> Subclause,
# instead of settling into whatever shape a physics simulation happens to
# find. This is what "aligning" the graph means here: a legal document has a
# real hierarchy, and the layout should look like one.
GROUP_LEVEL = {"part": 0, "article": 1, "clause": 2, "subclause": 3}
DIM_COLOR = "rgba(120,128,160,0.35)"


def render_graph_html(
    store: ConstitutionGraphStore,
    highlight_ids: set[str] | None = None,
    height: str = "560px",
    layout: str = "hierarchical",
) -> str:
    """Build a self-contained HTML string of the graph, dimming everything
    except `highlight_ids` (and their direct neighbours) when provided.

    layout="hierarchical" (default) lays the graph out top-down by legal
    depth (Part -> Article -> Clause -> Subclause), which is far more
    readable than an unconstrained physics layout once the graph has more
    than a couple dozen nodes. layout="physics" falls back to the old
    force-directed behaviour if you prefer an organic look.
    """

    highlight_ids = highlight_ids or set()
    focus_ids = set(highlight_ids)
    for nid in list(highlight_ids):
        if nid in store.graph:
            focus_ids.update(store.graph.predecessors(nid))
            focus_ids.update(store.graph.successors(nid))

    net = Network(height=height, width="100%", bgcolor="#0d1122", font_color="#eef1ff", directed=True)

    for node_id, data in store.graph.nodes(data=True):
        group = data["group"]
        is_focus = (not highlight_ids) or (node_id in focus_ids)
        is_hit = node_id in highlight_ids
        color = GROUP_COLORS.get(group, "#999999") if is_focus else DIM_COLOR
        size = GROUP_SIZES.get(group, 10) + (6 if is_hit else 0)
        border = "#ffffff" if is_hit else GROUP_COLORS.get(group, "#999999")
        node_kwargs = dict(
            label=data["label"] if len(data["label"]) < 40 else data["label"][:37] + "...",
            title=(data.get("title") or data["label"])[:300],
            color={"background": color, "border": border},
            size=size,
            shape="box" if group == "part" else ("diamond" if group == "subclause" else "dot"),
            font={"color": "#eef1ff" if is_focus else "rgba(200,205,230,0.25)", "size": 11},
        )
        if layout == "hierarchical":
            node_kwargs["level"] = GROUP_LEVEL.get(group, 3)
        net.add_node(node_id, **node_kwargs)

    for u, v, data in store.graph.edges(data=True):
        both_focus = (u in focus_ids) and (v in focus_ids)
        net.add_edge(
            u,
            v,
            color="#ffffff" if both_focus and highlight_ids else "rgba(150,155,190,0.18)",
            width=2 if both_focus and highlight_ids else 1,
        )

    if layout == "hierarchical":
        net.set_options("""
        {
          "interaction": {"hover": true, "tooltipDelay": 120},
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "UD",
              "sortMethod": "directed",
              "levelSeparation": 130,
              "nodeSpacing": 90,
              "treeSpacing": 140,
              "blockShifting": true,
              "edgeMinimization": true,
              "parentCentralization": true
            }
          },
          "physics": {
            "hierarchicalRepulsion": {"nodeDistance": 100},
            "solver": "hierarchicalRepulsion",
            "stabilization": {"iterations": 150}
          },
          "edges": {"smooth": {"type": "cubicBezier", "forceDirection": "vertical", "roundness": 0.4}}
        }
        """)
    else:
        net.barnes_hut(gravity=-6000, spring_length=90, spring_strength=0.04, damping=0.5)
        net.set_options("""
        {
          "interaction": {"hover": true, "tooltipDelay": 120},
          "physics": {"stabilization": {"iterations": 200}}
        }
        """)

    return net.generate_html(notebook=False)
