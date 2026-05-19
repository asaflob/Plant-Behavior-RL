"""Counterfactual decision DAG: branch best / 2nd / worst at every (state, day)."""

from collections import defaultdict, deque

import matplotlib.pyplot as plt


BRANCH_ROLES = ("best", "second", "worst")


def _pick_actions(agent, s, roles):
    actions = agent["alt_actions"].get(s)
    if not actions:
        return []
    chosen = []
    if "best" in roles:
        chosen.append(("best", actions[0]))
    if "second" in roles and len(actions) >= 2:
        chosen.append(("second", actions[1]))
    if "worst" in roles:
        chosen.append(("worst", actions[-1]))
    return chosen


def build_decision_dag(s0, agent, max_depth=5, roles=BRANCH_ROLES):
    """BFS from (s0, day=0); branch at every (state, day); merge on (state, day).

    Returns (nodes_set, edges_list). Each edge:
      {src: (state, day), dst: (state, day+1), role, action, advantage}.
    """
    nodes = {(s0, 0)}
    edges = []
    frontier = deque([(s0, 0)])

    opt_succ = agent["optimal_successor"]
    alt_step = agent["alt_first_step"]
    adv      = agent["advantage"]

    while frontier:
        s, day = frontier.popleft()
        if day >= max_depth:
            continue

        for role, a in _pick_actions(agent, s, roles):
            ns = opt_succ.get(s) if role == "best" else alt_step.get((s, a))
            if ns is None:
                continue

            edges.append({
                "src": (s, day), "dst": (ns, day + 1),
                "role": role, "action": a,
                "advantage": adv.get(s, {}).get(a, 0.0),
            })
            key = (ns, day + 1)
            if key not in nodes:
                nodes.add(key)
                frontier.append(key)

    return nodes, edges


def render_decision_dag(nodes, edges, ax=None, output_path=None,
                        state_labeler=None, role_colors=None):
    """Layered DAG plot: x = day, y = state position within that day's column."""
    role_colors = role_colors or {"best": "#2e7d32", "second": "#ef6c00", "worst": "#c62828"}
    state_labeler = state_labeler or (lambda s: ",".join(f"{v:g}" for v in s))

    by_day = defaultdict(list)
    for s, day in nodes:
        by_day[day].append(s)
    for day in by_day:
        by_day[day].sort()

    pos = {}
    for day, states in by_day.items():
        n = len(states)
        for i, s in enumerate(states):
            pos[(s, day)] = (day, i - (n - 1) / 2.0)

    max_col = max(len(v) for v in by_day.values())
    if ax is None:
        fig, ax = plt.subplots(figsize=(2 + 1.8 * len(by_day), max(4, 0.7 * max_col)))

    for e in edges:
        x1, y1 = pos[e["src"]]
        x2, y2 = pos[e["dst"]]
        ax.annotate("",
                    xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->",
                                    color=role_colors[e["role"]],
                                    lw=1.5, alpha=0.85))
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, f"a={e['action']}\nΔ={e['advantage']:.2f}",
                fontsize=7, color=role_colors[e["role"]],
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    for (s, day), (x, y) in pos.items():
        ax.scatter([x], [y], s=220, c="#eeeeee", edgecolors="black", zorder=3)
        ax.text(x, y - 0.28, state_labeler(s),
                fontsize=7, ha="center", va="top", zorder=4)

    ax.set_xticks(sorted(by_day.keys()))
    ax.set_xticklabels([f"day {d}" for d in sorted(by_day.keys())])
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Counterfactual decision DAG (best=green, 2nd=orange, worst=red)")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return ax
