"""Perishable DCL 점검 — DCL vs BSP*. (BSP-low-EW 는 후속.)"""
import os, sys, time, io
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dcl_paper import PerishableEnv, BaseStockPolicy, DCLTrainer, evaluate_policy


def main():
    t0 = time.time(); lines = []
    def log(s): print(s, flush=True); lines.append(s)

    env = PerishableEnv(lifetime=3, lead_time=1, holding_cost=1, penalty_cost=10,
                        waste_cost=7, mean_demand=4, fifo_ratio=1.0, dist="poisson")
    log(f"[env] Perishable ml=3 τ=1 h=1 p=10 w=7 mean=4 FIFO | I_max={env.I_max} m={env.m}")
    ev = dict(n_runs=200, horizon=1200, warmup=100, seed=2024)

    # 빠른 BSP* (이전 스윕서 S*=11)
    best = None
    for S in range(9, 14):
        m, _ = evaluate_policy(env, BaseStockPolicy(S), **ev)
        if best is None or m < best[1]: best = (S, m)
    bsp_S, bsp_m = best
    log(f"[휴리스틱] BSP* S={bsp_S} cost={bsp_m:.4f}")

    log("\n[DCL] N=800 M=250 H=40 L=80 n=3 w=6")
    trainer = DCLTrainer(env, BaseStockPolicy(env.I_max),
                         n=3, N=800, M=250, H=40, L=80, w=6, seed=0)
    policies = trainer.train(verbose=True)

    log("\n[평가]")
    log(f"  BSP*(S={bsp_S})  {bsp_m:.4f}")
    bb = None
    for i, pol in enumerate(policies):
        m, c = evaluate_policy(env, pol, **ev)
        log(f"  DCL gen{i+1}   {m:.4f}+-{c:.4f}  (vs BSP* {(m-bsp_m)/bsp_m*100:+.2f}%)")
        if bb is None or m < bb[1]: bb = (i+1, m)
    log(f"\n[결과] DCL best gen{bb[0]}={bb[1]:.4f} | BSP* 대비 {(bb[1]-bsp_m)/bsp_m*100:+.2f}% | {time.time()-t0:.0f}s")
    io.open(os.path.join(os.path.dirname(__file__), "..", "knowledge", "results_perishable.txt"),
            "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
