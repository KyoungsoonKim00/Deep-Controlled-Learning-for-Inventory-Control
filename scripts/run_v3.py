"""Plan A 검증 v3 — B2(I_max 캡) 수정 후. DCL 이 BSP*/CBS* 따라잡는지 확인."""
import os, sys, time, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dcl_paper import (LostSalesEnv, BaseStockPolicy, CappedBaseStockPolicy,
                       DCLTrainer, evaluate_policy)


def main():
    t0 = time.time(); lines = []
    def log(s): print(s, flush=True); lines.append(s)

    env = LostSalesEnv(3, 1, 9, 5, "poisson")
    log(f"[env] I_max={env.I_max} (수정후) m={env.m} num_actions={env.num_actions}")
    ev = dict(n_runs=200, horizon=1200, warmup=100, seed=2024)

    bsp = BaseStockPolicy(23); cbs = CappedBaseStockPolicy(24, 6)
    bsp_m, _ = evaluate_policy(env, bsp, **ev)
    cbs_m, _ = evaluate_policy(env, cbs, **ev)
    log(f"[휴리스틱] BSP*(23)={bsp_m:.4f} | CBS*(24,6)={cbs_m:.4f}")

    log("\n[DCL] N=800 M=250 H=40 L=80 n=3 w=10")
    trainer = DCLTrainer(env, BaseStockPolicy(env.I_max),
                         n=3, N=800, M=250, H=40, L=80, w=10, seed=0)
    policies = trainer.train(verbose=True)

    log("\n[평가]")
    best = None
    for i, pol in enumerate(policies):
        m, c = evaluate_policy(env, pol, **ev)
        log(f"  DCL gen{i+1}  {m:.4f}±{c:.4f}  (vs CBS* {(m-cbs_m)/cbs_m*100:+.2f}% | vs BSP* {(m-bsp_m)/bsp_m*100:+.2f}%)")
        if best is None or m < best[1]: best = (i+1, m)
    log(f"\n[결과] DCL best gen{best[0]}={best[1]:.4f} | CBS* 대비 {(best[1]-cbs_m)/cbs_m*100:+.2f}% | 시간 {time.time()-t0:.0f}s")
    io.open(os.path.join(os.path.dirname(__file__), "..", "knowledge", "results_v3.txt"),
            "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
