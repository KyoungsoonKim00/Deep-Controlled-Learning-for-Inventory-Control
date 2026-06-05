import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
from dcl_paper import (LostSalesEnv, BaseStockPolicy, Classifier, simulator,
                       collect_samples, evaluate_policy)

env = LostSalesEnv(3, 1, 9, 5, "poisson")
print(f"I_max={env.I_max} m={env.m} num_actions={env.num_actions}")

# 1) 라벨 분포 (BSP 초기정책으로 수집)
data = collect_samples(env, BaseStockPolicy(env.I_max), num_samples=300, H=30, M=60, L=60, seed=1)
labels = [a for _, a in data]
print("\n[라벨 분포] action -> count:", dict(sorted(Counter(labels).items())))
print(" 라벨 unique:", sorted(set(labels)))
# 상태 예시 몇 개
for st, a in data[:6]:
    print(f"  state={np.array(st).astype(int)} invpos={int(sum(st))} label={a}")

# 2) 시뮬레이션 기반 정책(NN 없이 simulator 직접)을 평가 → 라벨 품질 검증
class SimPolicy:
    def __init__(self, env, M, H, seed=0):
        self.env, self.M, self.H = env, M, H
        self.rng = np.random.default_rng(seed)
    def get_action(self, state, env):
        return simulator(self.env, state, BaseStockPolicy(env.I_max), self.M, self.H, self.rng)

ev_small = dict(n_runs=30, horizon=300, warmup=50, seed=2024)
bsp_m, _ = evaluate_policy(env, BaseStockPolicy(23), **ev_small)
sim_m, _ = evaluate_policy(env, SimPolicy(env, 60, 30), **ev_small)
print(f"\n[정책 평가 (소규모)] BSP(23)={bsp_m:.3f}  SimulatorPolicy={sim_m:.3f}")
print("  -> SimulatorPolicy 가 BSP 와 비슷하거나 낮으면 '라벨은 정상', 분류기가 범인.")

# 3) 분류기 학습 후 행동 맵
clf = Classifier(env.state_size, env.num_actions)
clf.train_policy(data, verbose=False)
print("\n[분류기 행동 맵] on-hand 변화 (pipeline=0,0):")
for oh in [0, 3, 6, 10, 15, 20]:
    s = np.array([oh, 0, 0])
    print(f"  state={s} invpos={oh} -> action={clf.get_action(s, env)}")
print(" pipeline 변화 (on-hand=2):")
for p in [0, 4, 8, 12]:
    s = np.array([2, p, 0])
    print(f"  state={s} invpos={2+p} -> action={clf.get_action(s, env)}")
