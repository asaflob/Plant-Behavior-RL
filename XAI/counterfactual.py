"""Counterfactual "what-if" trajectory: take action a from s0, then act optimally."""


def counterfactual_rollout(s0, a, alt_first_step, optimal_successor, max_steps=60):
    s1 = alt_first_step.get((s0, a))
    if s1 is None:
        return [s0]
    traj = [s0, s1]
    cur = s1
    for _ in range(max_steps - 1):
        nxt = optimal_successor.get(cur)
        if nxt is None or nxt == cur:
            break
        traj.append(nxt)
        cur = nxt
    return traj


def fan_trajectories(s0, agent, max_steps=60):
    """Yield (action, advantage, trajectory) for each alternative stored at s0.

    `agent` is the dict loaded from the pickle written by PlantGrowthTrainer.save_agent.
    """
    actions = agent["alt_actions"].get(s0, [])
    adv     = agent["advantage"].get(s0, {})
    for a in actions:
        traj = counterfactual_rollout(
            s0, a, agent["alt_first_step"], agent["optimal_successor"], max_steps
        )
        yield a, adv.get(a, 0.0), traj
