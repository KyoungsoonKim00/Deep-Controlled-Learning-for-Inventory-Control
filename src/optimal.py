"""
정확 최적해 — lost sales 평균비용 MDP를 relative value iteration 으로 푼다.
논문의 핵심 지표(optimality gap = (v_pi - v*)/v*)를 산출하기 위함.

상태공간: inventory position(sum) <= I_max 인 정수벡터 (유한). τ가 작은 소형 인스턴스 전용.
"""
import numpy as np
import scipy.stats as stats
from itertools import product


def enumerate_states(tau, I_max):
    states = []
    # 각 성분 0..I_max, 합 <= I_max
    def rec(prefix, remaining, depth):
        if depth == tau:
            states.append(tuple(prefix)); return
        for v in range(0, remaining + 1):
            rec(prefix + [v], remaining - v, depth + 1)
    rec([], I_max, 0)
    return states


def solve_lost_sales_optimal(env, demand_trunc=None, tol=1e-9, max_iter=20000):
    """relative value iteration (평균비용). 반환: (g* 최적 평균비용, policy dict)."""
    tau, h, p, I_max, m = env.tau, env.h, env.p, env.I_max, env.m
    mean = env.demand.mean
    if demand_trunc is None:
        demand_trunc = int(stats.poisson.ppf(0.99999, mean)) + 5 if env.demand.dist == "poisson" else int(mean * 8)
    ds = np.arange(0, demand_trunc + 1)
    if env.demand.dist == "poisson":
        pmf = stats.poisson.pmf(ds, mean); pmf[-1] += 1 - pmf.sum()
    else:
        pp = 1.0 / (mean + 1.0)
        pmf = stats.geom.pmf(ds + 1, pp); pmf[-1] += 1 - pmf.sum()

    states = enumerate_states(tau, I_max)
    idx = {s: i for i, s in enumerate(states)}
    nS = len(states)
    V = np.zeros(nS)

    def feasible(s):
        room = I_max - sum(s)
        return list(range(0, max(0, min(m, room)) + 1))

    def step_cost(s, d):
        return h * max(0, s[0] - d) + p * max(0, d - s[0])

    def next_state(s, a, d):
        if tau == 1:
            return (a,)
        ns0 = max(0, s[0] - d) + s[1]
        return (ns0,) + tuple(s[2:tau]) + (a,)

    ref = 0
    gain = 0.0
    for it in range(max_iter):
        Vn = np.empty(nS)
        for i, s in enumerate(states):
            best = np.inf
            for a in feasible(s):
                # 기대비용 + 기대 V'
                exp = 0.0
                for d, pr in zip(ds, pmf):
                    if pr <= 0: continue
                    exp += pr * (step_cost(s, d) + V[idx[next_state(s, a, d)]])
                if exp < best:
                    best = exp
            Vn[i] = best
        gain = Vn[ref]
        Vn -= Vn[ref]
        diff = np.max(np.abs(Vn - V))
        V = Vn
        if diff < tol:
            break

    # 최적 정책 추출
    policy = {}
    for s in states:
        best, ba = np.inf, 0
        for a in feasible(s):
            exp = 0.0
            for d, pr in zip(ds, pmf):
                if pr <= 0: continue
                exp += pr * (step_cost(s, d) + V[idx[next_state(s, a, d)]])
            if exp < best:
                best, ba = exp, a
        policy[s] = ba
    return gain, policy, it + 1


def policy_evaluate_lost_sales(env, base_stock_S, demand_trunc=None, tol=1e-9, max_iter=20000):
    """고정 BSP(S) 정책의 평균비용 가치 v_pi 와 정확한 one-step improvement pi_plus.
    pi_plus(s) = argmin_a E_d[C(s,a,d) + v_pi(f(s,a,d))]  — 논문 §5.2 정답기준.
    """
    tau, h, p, I_max, m = env.tau, env.h, env.p, env.I_max, env.m
    mean = env.demand.mean
    if demand_trunc is None:
        demand_trunc = int(stats.poisson.ppf(0.99999, mean)) + 5
    ds = np.arange(0, demand_trunc + 1)
    if env.demand.dist == "poisson":
        pmf = stats.poisson.pmf(ds, mean); pmf[-1] += 1 - pmf.sum()
    else:
        pp = 1.0 / (mean + 1.0); pmf = stats.geom.pmf(ds + 1, pp); pmf[-1] += 1 - pmf.sum()

    states = enumerate_states(tau, I_max)
    idx = {s: i for i, s in enumerate(states)}
    nS = len(states)

    def bsp_action(s):
        a = max(0, base_stock_S - sum(s))
        return int(min(a, m, I_max - sum(s)))

    def step_cost(s, d):
        return h * max(0, s[0] - d) + p * max(0, d - s[0])

    def next_state(s, a, d):
        if tau == 1: return (a,)
        return (max(0, s[0] - d) + s[1],) + tuple(s[2:tau]) + (a,)

    # 고정정책 가치반복 (평균비용, relative)
    V = np.zeros(nS); ref = 0
    for _ in range(max_iter):
        Vn = np.empty(nS)
        for i, s in enumerate(states):
            a = bsp_action(s)
            exp = 0.0
            for d, pr in zip(ds, pmf):
                if pr > 0: exp += pr * (step_cost(s, d) + V[idx[next_state(s, a, d)]])
            Vn[i] = exp
        g = Vn[ref]; Vn -= g
        if np.max(np.abs(Vn - V)) < tol: V = Vn; break
        V = Vn

    def feasible(s):
        room = I_max - sum(s); return list(range(0, max(0, min(m, room)) + 1))

    pi_plus = {}
    for s in states:
        best, ba = np.inf, 0
        for a in feasible(s):
            exp = 0.0
            for d, pr in zip(ds, pmf):
                if pr > 0: exp += pr * (step_cost(s, d) + V[idx[next_state(s, a, d)]])
            if exp < best: best, ba = exp, a
        pi_plus[s] = ba
    return pi_plus, states


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from dcl_paper import LostSalesEnv
    env = LostSalesEnv(3, 1, 9, 5, "poisson")
    g, pol, iters = solve_lost_sales_optimal(env)
    print(f"I_max={env.I_max} states={len(enumerate_states(env.tau, env.I_max))} iters={iters}")
    print(f"최적 평균비용 v* = {g:.4f}")
