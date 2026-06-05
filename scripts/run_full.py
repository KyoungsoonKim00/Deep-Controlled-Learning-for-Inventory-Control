"""Full 재현 — 정확 최적해(value iteration) 기준 optimality gap 보고. DCL 강한 예산."""
import os, sys, time, io, numpy as np
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dcl_paper import (LostSalesEnv, BaseStockPolicy, CappedBaseStockPolicy,
                       DCLTrainer, evaluate_policy)
from optimal import solve_lost_sales_optimal


class OptimalPolicy:
    def __init__(self, table): self.table = table
    def get_action(self, state, env):
        return self.table.get(tuple(int(x) for x in state), 0)


def main():
    t0 = time.time(); lines = []
    def log(s): print(s, flush=True); lines.append(s)

    env = LostSalesEnv(3, 1, 9, 5, "poisson")
    log(f"[env] LostSales Poisson mean=5 τ=3 h=1 p=9 | I_max={env.I_max} m={env.m}")

    log("\n[0] 정확 최적해 (relative value iteration)")
    vstar, opt_table, iters = solve_lost_sales_optimal(env)
    log(f"  v* = {vstar:.4f} (수렴 {iters} iters)")

    ev = dict(n_runs=400, horizon=2000, warmup=100, seed=2024)
    def gap(m): return (m - vstar) / vstar * 100

    log("\n[1] 기준 정책 평가 (avg cost/period, 400run x 2000period)")
    opt = OptimalPolicy(opt_table)
    opt_m, opt_ci = evaluate_policy(env, opt, **ev)
    bsp_m, _ = evaluate_policy(env, BaseStockPolicy(23), **ev)
    cbs_m, _ = evaluate_policy(env, CappedBaseStockPolicy(24, 6), **ev)
    log(f"  Optimal(sim) {opt_m:.4f}+-{opt_ci:.4f}  gap {gap(opt_m):+.2f}%  (검증: v*~={vstar:.4f})")
    log(f"  BSP*(23)     {bsp_m:.4f}  gap {gap(bsp_m):+.2f}%")
    log(f"  CBS*(24,6)   {cbs_m:.4f}  gap {gap(cbs_m):+.2f}%")

    log("\n[2] DCL 학습 (N=2000 M=500 H=40 L=100 n=3 w=12)")
    trainer = DCLTrainer(env, BaseStockPolicy(env.I_max),
                         n=3, N=2000, M=500, H=40, L=100, w=12, seed=0)
    policies = trainer.train(verbose=True)

    log("\n[3] DCL 평가 (optimality gap)")
    best = None
    for i, pol in enumerate(policies):
        m, c = evaluate_policy(env, pol, **ev)
        log(f"  DCL gen{i+1}  {m:.4f}±{c:.4f}  gap {gap(m):+.2f}%  (vs CBS* {(m-cbs_m)/cbs_m*100:+.2f}%)")
        if best is None or m < best[1]: best = (i+1, m)
    log(f"\n[결과] DCL best gen{best[0]}={best[1]:.4f} | optimality gap {gap(best[1]):+.2f}% | 논문 목표 ≤0.2%")
    log(f"[시간] {time.time()-t0:.0f}s")

    io.open(os.path.join(os.path.dirname(__file__), "..", "knowledge", "results_full.txt"),
            "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
