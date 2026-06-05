import os, sys, numpy as np
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dcl_paper import PerishableEnv, BaseStockPolicy, evaluate_policy

# Haijema 풍 소형: 수명3, 리드타임1, h=1, p=10, w=7, mean=4, FIFO
env = PerishableEnv(lifetime=3, lead_time=1, holding_cost=1, penalty_cost=10,
                    waste_cost=7, mean_demand=4, fifo_ratio=1.0, dist="poisson")
print(f"[env] ml={env.ml} tau={env.tau} I_max={env.I_max} m={env.m} "
      f"num_actions={env.num_actions} inv_size={env.inv_size}")

# feasibility 점검
for st in [np.zeros(env.inv_size, int), np.array([2,3,4,0][:env.inv_size])]:
    print(f"  state={st} invpos={int(st.sum())} feasible={env.feasible_actions(st)}")

ev = dict(n_runs=150, horizon=1000, warmup=100, seed=2024)

class ConstOrder:
    def __init__(self,c): self.c=c
    def get_action(self,s,env): return min(self.c, env.m)

print("\n[base-stock S 스윕] (최적 S* 가 I_max 미만이면 B2 캡 문제 없음)")
best=None
for S in range(4, env.I_max+1):
    m,_ = evaluate_policy(env, BaseStockPolicy(S), **ev)
    flag = " <- I_max" if S==env.I_max else ""
    if best is None or m<best[1]: best=(S,m)
    if S<=18 or S==env.I_max:
        print(f"  S={S}: {m:.4f}{flag}")
print(f"  => BSP* S={best[0]} cost={best[1]:.4f}  (I_max={env.I_max})")
print(f"  B2 캡 바인딩? {'YES (S*==I_max, 캡 풀어야)' if best[0]>=env.I_max else 'NO (여유 있음)'}")
