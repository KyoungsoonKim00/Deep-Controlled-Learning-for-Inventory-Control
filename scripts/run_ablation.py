"""
SH/CRN ablation (논문 §5.2) — 4 시뮬레이션 전략의 '정답 분류율' vs 예산 M.
정답 = 정확 one-step improvement pi_plus(s) (고정정책 BSP(S_pi) 하).
전략: SH+CRN(=DCL), SH-CRN, Uniform+CRN, Uniform-CRN(=DCL0). 총예산 B_s=M|A_s| 동일.
"""
import os, sys, time, io, math, numpy as np
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dcl_paper import LostSalesEnv, BaseStockPolicy, rollout
from optimal import policy_evaluate_lost_sales
from concurrent.futures import ProcessPoolExecutor, as_completed

ENV_ARGS = dict(lead_time=3, holding_cost=1, penalty_cost=9, mean_demand=5, dist="poisson")
S_PI = 23          # 연속정책 BSP base-stock (≈최적, 행동 구별 가장 어려움)
H = 40
M_GRID = [50, 100, 200, 500, 1000, 2000]
N_STATES = 20
REPS = 40
WORKERS = 12


def make_env():
    return LostSalesEnv(**ENV_ARGS)


def avg_costs(env, s, pi, action_list, scenarios):
    """각 action 의 평균 trajectory cost. scenarios: list of (per-action 또는 shared) 시나리오."""
    sums = {a: 0.0 for a in action_list}
    for sc in scenarios:  # sc: dict a->scenario 또는 단일 시나리오(shared)
        if isinstance(sc, dict):
            for a in action_list:
                sums[a] += rollout(env, s, a, pi, H, sc[a])
        else:
            for a in action_list:
                sums[a] += rollout(env, s, a, pi, H, sc)
    n = len(scenarios)
    return {a: sums[a] / n for a in action_list}


def gen(env, rng): return env.sample_scenario(H, rng)


def strat_uniform(env, s, pi, M, crn, rng):
    A = env.feasible_actions(s)
    if len(A) <= 1: return A[0] if A else 0
    if crn:
        scen = [gen(env, rng) for _ in range(M)]
        q = avg_costs(env, s, pi, A, scen)
    else:
        q = {}
        for a in A:
            tot = sum(rollout(env, s, a, pi, H, gen(env, rng)) for _ in range(M))
            q[a] = tot / M
    return min(q, key=q.get)


def strat_sh(env, s, pi, M, crn, rng):
    A = env.feasible_actions(s)
    if len(A) <= 1: return A[0] if A else 0
    B = M * len(A)
    rounds = max(1, math.ceil(math.log2(len(A))))
    acc = {a: 0.0 for a in A}; cnt = {a: 0 for a in A}
    Ar = list(A)
    for r in range(rounds):
        t_r = max(1, B // (len(Ar) * rounds))
        for _ in range(t_r):
            if crn:
                sc = gen(env, rng)
                for a in Ar:
                    acc[a] += rollout(env, s, a, pi, H, sc); cnt[a] += 1
            else:
                for a in Ar:
                    acc[a] += rollout(env, s, a, pi, H, gen(env, rng)); cnt[a] += 1
        avg = {a: acc[a] / cnt[a] for a in Ar}
        Ar = sorted(Ar, key=lambda a: avg[a])[:math.ceil(len(Ar) / 2)]
    return Ar[0]


STRATS = {
    "SH+CRN (DCL)":   lambda e, s, pi, M, rng: strat_sh(e, s, pi, M, True, rng),
    "SH-CRN":         lambda e, s, pi, M, rng: strat_sh(e, s, pi, M, False, rng),
    "Uniform+CRN":    lambda e, s, pi, M, rng: strat_uniform(e, s, pi, M, True, rng),
    "Uniform-CRN(DCL0)": lambda e, s, pi, M, rng: strat_uniform(e, s, pi, M, False, rng),
}


def worker(args):
    name, M, seed, states, truth = args
    env = make_env(); pi = BaseStockPolicy(S_PI); fn = STRATS[name]
    rng = np.random.default_rng(seed)
    hits = 0; tot = 0
    for s in states:
        sa = np.array(s)
        for _ in range(REPS):
            a = fn(env, sa, pi, M, rng)
            hits += int(a == truth[s]); tot += 1
    return name, M, hits, tot


def sample_states(env, pi, n, rng):
    """π 궤적에서 행동선택이 비자명한(>=3 feasible) 상태 n개."""
    states = []; s = env.get_initial_state()
    scen = env.sample_scenario(5000, rng)
    for t in range(5000):
        if len(env.feasible_actions(s)) >= 3:
            states.append(tuple(int(x) for x in s))
        a = pi.get_action(s, env); s = env.get_next_state(s, a, scen[t])
        if len(states) >= 400: break
    uniq = list(dict.fromkeys(states))
    rng.shuffle(uniq)
    return uniq[:n]


def main():
    t0 = time.time(); lines = []
    def log(x): print(x, flush=True); lines.append(x)

    env = make_env()
    log(f"[env] LostSales τ=3 p=9 mean=5 | I_max={env.I_max} m={env.m} | 연속정책 BSP(S={S_PI})")
    log("[정답기준] 정확 one-step improvement pi_plus 계산 중...")
    truth_full, _ = policy_evaluate_lost_sales(env, S_PI)

    rng = np.random.default_rng(0)
    states = sample_states(env, BaseStockPolicy(S_PI), N_STATES, rng)
    truth = {s: truth_full[s] for s in states}
    log(f"  테스트 상태 {len(states)}개, 전략 4종, M={M_GRID}, reps={REPS}")

    jobs = []
    sd = 100
    for name in STRATS:
        for M in M_GRID:
            jobs.append((name, M, sd, states, truth)); sd += 1

    results = {name: {} for name in STRATS}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(worker, j) for j in jobs]
        for f in as_completed(futs):
            name, M, hits, tot = f.result()
            results[name][M] = hits / tot

    log("\n[정답 분류율 %] (행=전략, 열=M)")
    header = "  " + "전략".ljust(20) + "".join(f"{M:>8}" for M in M_GRID)
    log(header)
    for name in STRATS:
        row = "  " + name.ljust(20) + "".join(f"{results[name][M]*100:>7.1f}%" for M in M_GRID)
        log(row)

    # 배수 추정: DCL(SH+CRN) M=250 근방 정확도를 DCL0 가 어느 M서 도달?
    dcl = results["SH+CRN (DCL)"]; dcl0 = results["Uniform-CRN(DCL0)"]
    ref_M = 200
    ref_acc = dcl[ref_M]
    reach = next((M for M in M_GRID if dcl0[M] >= ref_acc), None)
    log(f"\n[배수 하한] DCL(SH+CRN) M={ref_M} 정확도 {ref_acc*100:.1f}% 를 "
        f"DCL0 는 M={'>'+str(M_GRID[-1]) if reach is None else reach} 에서 도달 "
        f"→ 약 {('>'+str(M_GRID[-1]//ref_M) if reach is None else round(reach/ref_M,1))}배")
    log(f"[시간] {time.time()-t0:.0f}s")

    io.open(os.path.join(os.path.dirname(__file__), "..", "knowledge", "results_ablation.txt"),
            "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
