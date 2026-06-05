"""
Lost Sales 재현 실험 러너 (Plan A). DCL vs BSP* vs CBS* 평균비용 비교.
실행: py -3.12 scripts/run_lost_sales.py
결과: knowledge/results_lost_sales.txt 에도 저장.
"""
import os, sys, time, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dcl_paper import (LostSalesEnv, BaseStockPolicy, DCLTrainer,
                       evaluate_policy, optimize_base_stock, optimize_capped_base_stock)


def main():
    t0 = time.time()
    lines = []
    def log(s):
        print(s, flush=True); lines.append(s)

    # 논문 소형 인스턴스: Poisson mean=5, tau=3, h=1, p=9
    env = LostSalesEnv(lead_time=3, holding_cost=1, penalty_cost=9, mean_demand=5, dist="poisson")
    log(f"[env] LostSales Poisson mean=5 tau=3 h=1 p=9 | I_max={env.I_max} m={env.m} num_actions={env.num_actions}")

    ev = dict(n_runs=300, horizon=1500, warmup=100, seed=2024)

    log("\n[1] 휴리스틱 최적화")
    bsp, bi = optimize_base_stock(env, **ev)
    cbs, ci = optimize_capped_base_stock(env, **ev)
    log(f"  BSP*  S={bi[0]}            avg_cost={bi[1]:.4f}")
    log(f"  CBS*  S={ci[0]} cap={ci[1]}      avg_cost={ci[2]:.4f}")

    log("\n[2] DCL 학습 (N=800, M=60, H=30, L=60, n=3, w=10)")
    trainer = DCLTrainer(env, BaseStockPolicy(env.I_max),
                         n=3, N=800, M=60, H=30, L=60, w=10, seed=0)
    policies = trainer.train(verbose=True)

    log("\n[3] 평가 비교 (avg cost/period, 300 run x 1500 period)")
    bsp_m, bsp_ci = evaluate_policy(env, bsp, **ev)
    cbs_m, cbs_ci = evaluate_policy(env, cbs, **ev)
    log(f"  BSP*        {bsp_m:.4f} +- {bsp_ci:.4f}")
    log(f"  CBS*        {cbs_m:.4f} +- {cbs_ci:.4f}   (vs BSP* {(cbs_m-bsp_m)/bsp_m*100:+.2f}%)")
    best = None
    for i, pol in enumerate(policies):
        m, c = evaluate_policy(env, pol, **ev)
        gap_cbs = (m - cbs_m) / cbs_m * 100
        gap_bsp = (m - bsp_m) / bsp_m * 100
        log(f"  DCL gen{i+1}   {m:.4f} +- {c:.4f}   (vs CBS* {gap_cbs:+.2f}% | vs BSP* {gap_bsp:+.2f}%)")
        if best is None or m < best[1]:
            best = (i + 1, m)
    log(f"\n[결과] DCL best = gen{best[0]} ({best[1]:.4f}). "
        f"CBS* 대비 {(best[1]-cbs_m)/cbs_m*100:+.2f}%, BSP* 대비 {(best[1]-bsp_m)/bsp_m*100:+.2f}%.")
    log(f"[시간] 총 {time.time()-t0:.1f}s")

    out = os.path.join(os.path.dirname(__file__), "..", "knowledge", "results_lost_sales.txt")
    with io.open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("saved:", os.path.abspath(out))


if __name__ == "__main__":
    main()
