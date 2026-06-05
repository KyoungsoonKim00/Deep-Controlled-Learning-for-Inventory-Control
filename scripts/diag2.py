import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dcl_paper import LostSalesEnv, BaseStockPolicy, simulator, rollout

env = LostSalesEnv(3, 1, 9, 5, "poisson")
bsp = BaseStockPolicy(env.I_max)
print(f"I_max={env.I_max} m={env.m}")

def true_q(state, H, reps, seed=0):
    """각 feasible 행동의 참 기대 H-비용 (독립 rollout 평균, BSP 연속)."""
    rng = np.random.default_rng(seed)
    feas = env.feasible_actions(state)
    q = {}
    for a in feas:
        c = 0.0
        for _ in range(reps):
            scen = env.sample_scenario(H, rng)
            c += rollout(env, state, a, bsp, H, scen)
        q[a] = c / reps
    return q

for st in [np.array([0,0,0]), np.array([2,0,0]), np.array([5,5,4])]:
    for H in [30, 60]:
        q = true_q(st, H, reps=4000, seed=1)
        best = min(q, key=q.get)
        bsp_a = bsp.get_action(st, env)
        print(f"\nstate={st} invpos={int(st.sum())} H={H}")
        print("  true q(a):", {a: round(v,1) for a,v in q.items()})
        print(f"  true argmin={best} | BSP picks={bsp_a}")
        # simulator 선택 (M별 5회)
        for M in [60, 500]:
            rng = np.random.default_rng(0)
            picks = [simulator(env, st, bsp, M, H, rng) for _ in range(5)]
            print(f"  simulator M={M}: {picks}")
