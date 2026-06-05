"""
dcl_paper.py — Deep Controlled Learning, 논문 충실 재현 (Plan A: 정상 i.i.d. 수요)

원논문: Temizöz et al., "Deep Controlled Learning for Inventory Control",
        arXiv:2011.15122 (§3 알고리즘, Appendix A MDP-EI 정식화).

기존 src/dcl_definitions.py 와의 차이 (진단 knowledge/diagnosis_repo_vs_paper.md 대응):
  - 요일(주간 계절) 제거 → 정상 i.i.d. 수요 (논문 가정).            [H3]
  - 시뮬레이터 CRN 시나리오 요일 미정렬 버그 소멸.                  [C1]
  - 평가 하니스(evaluate_policy) + 휴리스틱(BSP, CBS) 벤치마크 포함. [C2]
  - 행동집합 Aₛ 를 inventory position ≤ I_max 로 제약 + 분류기 마스킹. [H1, H2]
  - SH 선택을 누적합이 아닌 '평균'(누적/시나리오수)으로 명시.        [M1]
  - 하이퍼파라미터 기본값을 논문 근처로.                            [C3]
"""

from __future__ import annotations
import math
import numpy as np
from abc import ABC, abstractmethod
import scipy.stats as stats

# torch 는 분류기(Classifier)/DCLTrainer 에만 필요 → 지연 import.
# 환경·simulator·평가 하니스·휴리스틱은 numpy/scipy 만으로 동작한다.
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _TORCH = False
    device = None


# ===================================================================
# 수요 분포 (정상 i.i.d.)
# ===================================================================
class Demand:
    """평균 mean 의 i.i.d. 수요. dist in {'poisson','geometric'}."""
    def __init__(self, mean, dist="poisson", rng=None):
        self.mean = mean
        self.dist = dist
        self.rng = rng or np.random.default_rng()

    def sample(self, size):
        if self.dist == "poisson":
            return self.rng.poisson(self.mean, size=size)
        elif self.dist == "geometric":
            # 평균 mean 인 기하분포 (지지 0,1,2,...): p = 1/(mean+1)
            p = 1.0 / (self.mean + 1.0)
            return self.rng.geometric(p, size=size) - 1
        raise ValueError(self.dist)

    def ppf(self, q):
        """newsvendor 분위수 (정수)."""
        if self.dist == "poisson":
            return int(stats.poisson.ppf(q, self.mean))
        elif self.dist == "geometric":
            p = 1.0 / (self.mean + 1.0)
            return int(stats.geom.ppf(q, p) - 1)
        raise ValueError(self.dist)


# ===================================================================
# 환경 (MDP-EI, 정상 수요)
# ===================================================================
class BaseInventoryEnv(ABC):
    state_size: int
    num_actions: int

    @abstractmethod
    def get_initial_state(self): ...
    @abstractmethod
    def feasible_actions(self, state): ...
    @abstractmethod
    def get_next_state(self, state, action, demand): ...
    @abstractmethod
    def get_cost(self, state, action, demand): ...
    @abstractmethod
    def sample_scenario(self, length, rng=None): ...


class LostSalesEnv(BaseInventoryEnv):
    """
    이산시간 lost sales (Zipkin 2008). 상태 = [on-hand, pipeline_1..pipeline_{tau-1}].
    전이 f: s' = ((s0-d)^+ + s1, s2, ..., s_{tau-1}, a)
    비용 C = h*(s0-d)^+ + p*(d-s0)^+
    """
    def __init__(self, lead_time, holding_cost, penalty_cost, mean_demand,
                 dist="poisson", rng=None):
        assert lead_time >= 1
        self.tau = lead_time
        self.h, self.p = holding_cost, penalty_cost
        self.demand = Demand(mean_demand, dist, rng)
        crit = self.p / (self.p + self.h)               # newsvendor critical ratio
        # [B2 수정] inventory position 캡 I_max.
        # τ기간 newsvendor(=리뷰주기 0 가정)는 lost sales 최적 base-stock 을 과소추정해
        # feasibility 캡이 최적 정책을 막았음(DCL 이 BSP* 못 이기는 원인).
        # 리뷰주기+리드타임 = (τ+1) 기간 수요로 캡을 잡고 여유 마진을 둔다.
        self.I_max = max(1, self._cum_ppf(crit, self.tau + 1))
        self.m = max(1, self.demand.ppf(crit))          # 1기간 최대 주문량 (newsvendor bound)
        self.state_size = self.tau
        self.num_actions = self.m + 1

    def _cum_ppf(self, q, periods):
        """periods 기간 누적수요의 q-분위수 (정수)."""
        if self.demand.dist == "poisson":
            return int(stats.poisson.ppf(q, self.demand.mean * periods))
        s = self.demand.sample(size=(40000, periods)).sum(axis=1)
        return int(np.quantile(s, q))

    def get_initial_state(self):
        return np.zeros(self.tau, dtype=int)

    def inventory_position(self, state):
        return int(np.sum(state))

    def feasible_actions(self, state):
        # inventory position 이 I_max 를 넘지 않도록 + 단일주문 상한 m
        room = self.I_max - self.inventory_position(state)
        hi = max(0, min(self.m, room))
        return list(range(0, hi + 1))

    def sample_scenario(self, length, rng=None):
        d = Demand(self.demand.mean, self.demand.dist, rng) if rng is not None else self.demand
        return d.sample(size=length)

    def get_next_state(self, state, action, demand):
        s = np.asarray(state, dtype=int)
        sp = np.zeros(self.tau, dtype=int)
        if self.tau == 1:
            # 주문이 즉시(다음기) 가용: on-hand 이월 없음, action 이 다음 on-hand
            sp[0] = action
            return sp
        sp[0] = max(0, s[0] - demand) + s[1]
        if self.tau > 2:
            sp[1:self.tau - 1] = s[2:self.tau]
        sp[self.tau - 1] = action
        return sp

    def get_cost(self, state, action, demand):
        s0 = int(state[0])
        return self.h * max(0, s0 - demand) + self.p * max(0, demand - s0)


class PerishableEnv(BaseInventoryEnv):
    """
    부패성 재고 (Haijema 2019, 정상수요판). 상태 = [age_1..age_ml, pipeline_1..pipeline_{tau-1}].
    age_1 = 잔여수명 1기(다음기 폐기 임박) ... age_ml = 잔여수명 ml기(최신).
    수요는 FIFO 비율 f 로 분할. 비용 = w*폐기 + h*잔여 + p*품절.
    """
    def __init__(self, lifetime, lead_time, holding_cost, penalty_cost, waste_cost,
                 mean_demand, fifo_ratio=1.0, dist="poisson", rng=None):
        self.ml, self.tau = lifetime, lead_time
        self.h, self.p, self.w = holding_cost, penalty_cost, waste_cost
        self.fifo_ratio = fifo_ratio
        self.demand = Demand(mean_demand, dist, rng)
        crit = self.p / (self.p + self.h + self.w)
        # 주문 상한·I_max: tau+ml 기간 수요 newsvendor
        self.I_max = max(1, int(stats.poisson.ppf(crit, mean_demand * (self.tau + self.ml)))
                         if dist == "poisson" else 1)
        self.m = max(1, self.demand.ppf(self.p / (self.p + self.h)))
        self.inv_size = self.ml + max(self.tau - 1, 0)
        self.state_size = self.inv_size
        self.num_actions = self.m + 1
        self.rng = rng or np.random.default_rng()

    def get_initial_state(self):
        return np.zeros(self.inv_size, dtype=int)

    def inventory_position(self, state):
        return int(np.sum(state))

    def feasible_actions(self, state):
        room = self.I_max - self.inventory_position(state)
        hi = max(0, min(self.m, room))
        return list(range(0, hi + 1))

    def sample_scenario(self, length, rng=None):
        r = rng or self.rng
        total = (r.poisson(self.demand.mean, size=length) if self.demand.dist == "poisson"
                 else self.demand.sample(size=length))
        fifo = r.binomial(total, self.fifo_ratio)
        lifo = total - fifo
        return list(zip(fifo.tolist(), lifo.tolist()))

    def _sell(self, on_hand, demand_tuple):
        d_fifo, d_lifo = demand_tuple
        inv = np.array(on_hand, dtype=int)
        total = int(inv.sum())
        # FIFO: 오래된 것(낮은 인덱스 = 잔여수명 적음)부터
        rem = d_fifo
        for i in range(self.ml):
            sell = min(rem, inv[i]); inv[i] -= sell; rem -= sell
        # LIFO: 최신(높은 인덱스)부터
        rem = d_lifo
        for i in range(self.ml - 1, -1, -1):
            sell = min(rem, inv[i]); inv[i] -= sell; rem -= sell
        shortage = max(0, (d_fifo + d_lifo) - total)
        return inv, shortage

    def get_next_state(self, state, action, demand_tuple):
        s = np.asarray(state, dtype=int)
        on_hand = s[:self.ml]
        pipeline = s[self.ml:] if self.tau > 1 else np.array([], dtype=int)
        inv_after, _ = self._sell(on_hand, demand_tuple)
        sp = np.zeros(self.inv_size, dtype=int)
        # 노화: 잔여수명 i+1 → i. 인덱스0(잔여1기)은 판매후 남으면 폐기되어 사라짐.
        sp[:self.ml - 1] = inv_after[1:self.ml]
        arriving = action if self.tau <= 1 else pipeline[0]
        sp[self.ml - 1] += arriving
        if self.tau > 2:
            sp[self.ml:self.inv_size - 1] = pipeline[1:]
        if self.tau > 1:
            sp[self.inv_size - 1] = action
        return sp

    def get_cost(self, state, action, demand_tuple):
        on_hand = np.asarray(state[:self.ml], dtype=int)
        inv_after, shortage = self._sell(on_hand, demand_tuple)
        perished = inv_after[0]               # 잔여수명 1기 중 안 팔린 것 = 폐기
        leftover = int(inv_after[1:].sum())
        return self.w * perished + self.h * leftover + self.p * shortage


# ===================================================================
# 정책
# ===================================================================
class BaseStockPolicy:
    """주문 = clip(S - inventory_position, 0, m)."""
    def __init__(self, base_stock_level):
        self.S = base_stock_level

    def get_action(self, state, env):
        inv_pos = int(np.sum(state))
        a = max(0, self.S - inv_pos)
        return int(min(a, env.m))


class CappedBaseStockPolicy:
    """주문 = min(cap, clip(S - inventory_position, 0, m))."""
    def __init__(self, base_stock_level, cap):
        self.S, self.cap = base_stock_level, cap

    def get_action(self, state, env):
        inv_pos = int(np.sum(state))
        a = max(0, self.S - inv_pos)
        return int(min(a, self.cap, env.m))


_NNBase = nn.Module if _TORCH else object


class Classifier(_NNBase):
    """4층 MLP. forward → 행동 로짓. get_action 에서 불가행동 마스킹. (torch 필요)"""
    def __init__(self, input_size, num_actions):
        if not _TORCH:
            raise ImportError("Classifier/DCLTrainer 는 torch 가 필요합니다. `pip install torch` 후 사용.")
        super().__init__()
        self.num_actions = num_actions
        self.network = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        return self.network(x)

    def get_action(self, state, env):
        st = torch.as_tensor(np.asarray(state, dtype=np.float32), device=device).unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(st).squeeze(0).cpu().numpy()
        feas = env.feasible_actions(state)            # [H2] 마스킹
        if not feas:
            return 0
        masked = np.full(self.num_actions, -np.inf)
        for a in feas:
            if a < self.num_actions:
                masked[a] = logits[a]
        return int(np.argmax(masked))

    def train_policy(self, dataset_K, epochs=100, batch_size=64, lr=1e-3, patience=8, verbose=False):
        if len(dataset_K) < 5:
            if verbose: print("데이터셋 과소 — 학습 skip")
            return self
        states, actions = zip(*dataset_K)
        X = torch.as_tensor(np.array(states, dtype=np.float32))
        y = torch.as_tensor(np.array(actions, dtype=np.int64))
        n = len(X)
        n_val = max(1, int(0.2 * n))
        ds = TensorDataset(X, y)
        tr, va = torch.utils.data.random_split(ds, [n - n_val, n_val])
        tl = DataLoader(tr, batch_size=batch_size, shuffle=True)
        vl = DataLoader(va, batch_size=batch_size)
        opt = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        best, wait = float("inf"), 0
        for ep in range(epochs):
            self.train()
            for bx, by in tl:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                loss = loss_fn(self.forward(bx), by)
                loss.backward(); opt.step()
            self.eval(); vloss = 0.0
            with torch.no_grad():
                for bx, by in vl:
                    bx, by = bx.to(device), by.to(device)
                    vloss += loss_fn(self.forward(bx), by).item()
            vloss /= max(1, len(vl))
            if verbose: print(f"  epoch {ep+1}: val_loss={vloss:.4f}")
            if vloss < best - 1e-5:
                best, wait = vloss, 0
            else:
                wait += 1
                if wait >= patience:
                    break
        return self


# ===================================================================
# DCL (Algorithm 1–3) — SH + CRN
# ===================================================================
def sample_start_state(env, policy, L, rng):
    state = env.get_initial_state()
    scen = env.sample_scenario(L, rng)
    for j in range(L):
        a = policy.get_action(state, env)
        state = env.get_next_state(state, a, scen[j])
    return state


def rollout(env, start_state, initial_action, policy, H, scenario):
    """절단 궤적 누적비용. t=0 은 initial_action, 이후 policy (alpha=1)."""
    total = env.get_cost(start_state, initial_action, scenario[0])
    state = env.get_next_state(start_state, initial_action, scenario[0])
    for t in range(1, H):
        a = policy.get_action(state, env)
        total += env.get_cost(state, a, scenario[t])
        state = env.get_next_state(state, a, scenario[t])
    return total


def simulator(env, state, policy, M, H, rng):
    """
    Algorithm 3: SH + CRN. 상태 state 의 simulation-based action 반환.
    - CRN: 라운드 내 t_r 시나리오를 생존 행동 모두에 공유.
    - stockpiling: 누적합 유지, 랭킹은 평균(누적/시나리오수)으로.
    """
    A_s = env.feasible_actions(state)
    if len(A_s) <= 1:
        return A_s[0] if A_s else 0
    B_s = M * len(A_s)
    num_rounds = max(1, math.ceil(math.log2(len(A_s))))
    acc = {a: 0.0 for a in A_s}      # 누적 비용 합 (stockpiling)
    cnt = {a: 0 for a in A_s}        # 누적 시나리오 수
    A_r = list(A_s)
    for r in range(num_rounds):
        t_r = max(1, math.floor(B_s / (len(A_r) * num_rounds)))
        for _ in range(t_r):
            scen = env.sample_scenario(H, rng)          # CRN: 한 시나리오를 전 행동 공유
            for a in A_r:
                acc[a] += rollout(env, state, a, policy, H, scen)
            for a in A_r:
                cnt[a] += 1
        if len(A_r) > 1:
            avg = {a: acc[a] / cnt[a] for a in A_r}     # [M1] 평균으로 비교
            keep = math.ceil(len(A_r) / 2)
            A_r = sorted(A_r, key=lambda a: avg[a])[:keep]
    return A_r[0]


def collect_samples(env, policy, num_samples, H, M, L, seed):
    rng = np.random.default_rng(seed)
    samples = []
    state = sample_start_state(env, policy, L, rng)
    trans = env.sample_scenario(num_samples, rng)
    for k in range(num_samples):
        a_star = simulator(env, state, policy, M, H, rng)
        samples.append((np.asarray(state, dtype=float).tolist(), a_star))
        state = env.get_next_state(state, a_star, trans[k])
    return samples


class DCLTrainer:
    def __init__(self, env, initial_policy, n=3, N=5000, M=1000, H=40, L=100, w=1, seed=0):
        self.env, self.policy = env, initial_policy
        self.n, self.N, self.M, self.H, self.L, self.w, self.seed = n, N, M, H, L, w, seed

    def train(self, verbose=True):
        policies = []
        for i in range(self.n):
            if verbose:
                print(f"\n=== 근사 정책 반복 {i+1}/{self.n} (N={self.N}, M={self.M}, H={self.H}) ===")
            per_worker = math.ceil(self.N / max(1, self.w))
            dataset = []
            if self.w <= 1:
                dataset = collect_samples(self.env, self.policy, per_worker,
                                          self.H, self.M, self.L, self.seed + i * 1000)
            else:
                from concurrent.futures import ProcessPoolExecutor, as_completed
                with ProcessPoolExecutor(max_workers=self.w) as ex:
                    futs = [ex.submit(collect_samples, self.env, self.policy, per_worker,
                                      self.H, self.M, self.L, self.seed + i * 1000 + j)
                            for j in range(self.w)]
                    for f in as_completed(futs):
                        dataset.extend(f.result())
            if verbose:
                print(f"  샘플 {len(dataset)}개 수집 → 분류기 학습")
            # [B1 수정] 반복마다 새 분류기 (논문 π_{i+1}=Classifier(K_i)).
            # 이전 append(clf) 는 동일 객체를 n번 담아 세대 비교가 불가능했음.
            clf = Classifier(self.env.state_size, self.env.num_actions).to(device)
            clf.train_policy(dataset, verbose=False)
            self.policy = clf            # 다음 반복의 데이터 수집 연속정책 π_{i+1}
            policies.append(clf)
        return policies


# ===================================================================
# 평가 하니스 [C2] + 휴리스틱 최적화
# ===================================================================
def evaluate_policy(env, policy, n_runs=200, horizon=2000, warmup=100, seed=12345):
    """평균비용/기간 + 95% CI 반폭. 독립 run 의 평균비용으로 추정."""
    rng = np.random.default_rng(seed)
    run_costs = []
    for _ in range(n_runs):
        state = env.get_initial_state()
        scen = env.sample_scenario(warmup + horizon, rng)
        for t in range(warmup):
            a = policy.get_action(state, env)
            state = env.get_next_state(state, a, scen[t])
        total = 0.0
        for t in range(warmup, warmup + horizon):
            a = policy.get_action(state, env)
            total += env.get_cost(state, a, scen[t])
            state = env.get_next_state(state, a, scen[t])
        run_costs.append(total / horizon)
    arr = np.array(run_costs)
    mean = arr.mean()
    ci = 1.96 * arr.std(ddof=1) / math.sqrt(len(arr))
    return mean, ci


def optimize_base_stock(env, S_grid=None, **eval_kw):
    if S_grid is None:
        S_grid = range(max(1, env.I_max - 8), env.I_max + 9)
    best = None
    for S in S_grid:
        m, _ = evaluate_policy(env, BaseStockPolicy(S), **eval_kw)
        if best is None or m < best[1]:
            best = (S, m)
    return BaseStockPolicy(best[0]), best


def optimize_capped_base_stock(env, S_grid=None, cap_grid=None, **eval_kw):
    if S_grid is None:
        S_grid = range(max(1, env.I_max - 6), env.I_max + 7)
    if cap_grid is None:
        cap_grid = range(1, env.m + 1)
    best = None
    for S in S_grid:
        for cap in cap_grid:
            m, _ = evaluate_policy(env, CappedBaseStockPolicy(S, cap), **eval_kw)
            if best is None or m < best[2]:
                best = (S, cap, m)
    return CappedBaseStockPolicy(best[0], best[1]), best


# ===================================================================
# 데모: lost sales (논문 소형 인스턴스 근처)
# ===================================================================
if __name__ == "__main__":
    print("Lost Sales 재현 데모 (정상 Poisson 수요, mean=5, tau=3, p=9, h=1)")
    env = LostSalesEnv(lead_time=3, holding_cost=1, penalty_cost=9, mean_demand=5, dist="poisson")
    print(f"  I_max={env.I_max}, m(max order)={env.m}, num_actions={env.num_actions}")

    ev = dict(n_runs=100, horizon=2000, warmup=100, seed=7)

    print("\n[1] 휴리스틱 최적화 중...")
    bsp, bsp_info = optimize_base_stock(env, **ev)
    cbs, cbs_info = optimize_capped_base_stock(env, **ev)
    print(f"  BSP*  S={bsp_info[0]}        avg_cost={bsp_info[1]:.4f}")
    print(f"  CBS*  S={cbs_info[0]} cap={cbs_info[1]}  avg_cost={cbs_info[2]:.4f}")

    if not _TORCH:
        print("\n[2] torch 미설치 → DCL 학습 skip (휴리스틱 벤치마크까지만 검증).")
        print("    DCL 실행하려면 torch 가 있는 환경에서: trainer.train() 후 evaluate_policy.")
        raise SystemExit(0)

    print("\n[2] DCL 학습 (빠른 데모 설정; 논문값은 N=5000,M=1000,H=40,n=3)")
    trainer = DCLTrainer(env, BaseStockPolicy(env.I_max),
                         n=3, N=800, M=80, H=40, L=50, w=1, seed=0)
    policies = trainer.train(verbose=True)

    print("\n[3] 평가 비교")
    bsp_m, bsp_ci = evaluate_policy(env, bsp, **ev)
    cbs_m, cbs_ci = evaluate_policy(env, cbs, **ev)
    print(f"  BSP*           {bsp_m:.4f} ± {bsp_ci:.4f}")
    print(f"  CBS*           {cbs_m:.4f} ± {cbs_ci:.4f}")
    for i, pol in enumerate(policies):
        m, ci = evaluate_policy(env, pol, **ev)
        gap = (m - cbs_m) / cbs_m * 100
        print(f"  DCL gen{i+1}      {m:.4f} ± {ci:.4f}   (vs CBS* {gap:+.2f}%)")
