"""Plan A 검증 v2 — B1 수정 후, 예산 상향(M=400,H=40,N=800,n=3). 휴리스틱은 고정값 평가."""
import os, sys, time, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dcl_paper import (LostSalesEnv, BaseStockPolicy, CappedBaseStockPolicy,
                       DCLTrainer, evaluate_policy)


def main():
    t0 = time.time(); lines = []
    def log(s): print(s, flush=True); lines.append(s)

    env = LostSalesEnv(3, 1, 9, 5, "poisson")
    log(f"[env] LostSales Poisson mean=5 tau=3 h=1 p=9 | I_max={env.I_max} m={env.m}")
    ev = dict(n_runs=200, horizon=1200, warmup=100, seed=2024)

    bsp = BaseStockPolicy(23); cbs = CappedBaseStockPolicy(24, 6)
    bsp_m, bsp_ci = evaluate_policy(env, bsp, **ev)
    cbs_m, cbs_ci = evaluate_policy(env, cbs, **ev)
    log(f"\n[휴리스틱] BSP*(S=23) {bsp_m:.4f}±{bsp_ci:.4f} | CBS*(S=24,cap=6) {cbs_m:.4f}±{cbs_ci:.4f}")

    log("\n[DCL] N=800 M=400 H=40 L=80 n=3 w=10 (B1 수정: 세대별 새 분류기)")
    trainer = DCLTrainer(env, BaseStockPolicy(env.I_max),
                         n=3, N=800, M=400, H=40, L=80, w=10, seed=0)
    policies = trainer.train(verbose=True)

    log("\n[평가]")
    log(f"  BSP*      {bsp_m:.4f}")
    log(f"  CBS*      {cbs_m:.4f}")
    best = None
    for i, pol in enumerate(policies):
        m, c = evaluate_policy(env, pol, **ev)
        log(f"  DCL gen{i+1}  {m:.4f}±{c:.4f}  (vs CBS* {(m-cbs_m)/cbs_m*100:+.2f}% | vs BSP* {(m-bsp_m)/bsp_m*100:+.2f}%)")
        if best is None or m < best[1]: best = (i+1, m)
    log(f"\n[결과] DCL best gen{best[0]}={best[1]:.4f} | CBS* 대비 {(best[1]-cbs_m)/cbs_m*100:+.2f}%")
    log(f"[시간] {time.time()-t0:.1f}s")

    out = os.path.join(os.path.dirname(__file__), "..", "knowledge", "results_v2.txt")
    io.open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("saved:", os.path.abspath(out))


if __name__ == "__main__":
    main()
