import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dcl_paper import LostSalesEnv, BaseStockPolicy, evaluate_policy

env = LostSalesEnv(3, 1, 9, 5, "poisson")
print(f"I_max={env.I_max} m={env.m}")

class ConstOrder:
    def __init__(self, c): self.c = c
    def get_action(self, state, env): return min(self.c, env.m)

ev = dict(n_runs=200, horizon=1200, warmup=100, seed=2024)
print("\n[상수주문 정책]")
for c in range(3, 9):
    m, _ = evaluate_policy(env, ConstOrder(c), **ev)
    print(f"  order={c}: {m:.4f}")
print("\n[base-stock S 스윕]")
for S in range(14, 27):
    m, _ = evaluate_policy(env, BaseStockPolicy(S), **ev)
    print(f"  S={S}: {m:.4f}")
