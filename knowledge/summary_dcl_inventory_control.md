# Deep Controlled Learning for Inventory Control — 논문 요약

> arXiv:2011.15122 (TeX 소스 최종수정 2025-06). Temizöz, Imdahl, Dijkman, Lamghari-Idrissi, van Jaarsveld (TU Eindhoven). 코드: [DynaPlex](https://github.com/tarkantemizoz/DynaPlex).
>
> 이 요약은 본 repo(`Deep-Controlled-Learning-for-Inventory-Control`)의 구현이 논문 어디에 대응하는지, 무엇이 갭인지를 함께 기록한다.

---

## 1. 한 줄 요약

재고관리의 **높은 외생적 확률성**(수요·리드타임)에 맞춰 설계한 DRL 알고리즘. **근사 정책 반복(API)을 분류 문제로 환원**(CBPI)하고, **Sequential Halving(SH) + Common Random Numbers(CRN)**로 시뮬레이션 효율을 끌어올려, lost sales·perishable·random lead time 세 문제 전부에서 최고 휴리스틱을 능가(최적해 오차 ≤ 0.2%, 동일 하이퍼파라미터).

---

## 2. 문제 의식 (왜 범용 DRL이 재고에서 실패하나)

- 재고 MDP는 **MDP with Exogenous Inputs (MDP-EI)** — 의사결정과 무관한 독립 확률원(수요)이 비용에 큰 노이즈를 주입.
- A3C·PPO 등은 (1) 현 정책 궤적으로만 업데이트해 **상태 재방문이 적고**, (2) 신경망으로 **비용을 근사**하는데 외생 노이즈가 그 근사를 오염 → 행동 가치 비교가 부정확.
- 결과: 기존 DRL은 lost sales에서 **capped base-stock조차 못 이김**(A3C 최적오차 3–6%).

> 핵심 처방: 상태를 **여러 외생 시나리오로 평가**해야 행동 가치를 제대로 비교 가능 (Dietterich 2018). 단, 시나리오 다량 생성은 비싸다 → SH+CRN으로 해결.

---

## 3. MDP-EI 정식화

튜플 $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{W}, \Xi, f, C, \alpha, s_0 \rangle$.

- $\xi_t \sim \Xi$: 행동 실행 후 관측되는 **외생 입력**(i.i.d., 상태·행동과 독립). 예: 수요.
- 전이 $s' = f(s,a,\xi)$, 비용 $c = C(s,a,\xi)$ 모두 외생 입력에 의존.
- $\alpha = 1$ (평균비용 기준)을 재고문제에 사용.
- 효율적으로 $\xi$ 표본 생성 가능하다고 가정.

---

## 4. 알고리즘 (Algorithm 1–4)

### 4.1 근사 정책 반복 (API) — §3.1
- 정책 반복: $\pi^+(s) = \arg\min_a q_\pi(s,a)$. 정확 계산은 대규모 $\mathcal{S}$에서 불가.
- **Rollout 시뮬레이션**(Tesauro)으로 행동가치 추정: 궤적을 $H$ 스텝에서 절단, 외생 시나리오 $\boldsymbol\xi$ 하나당 $\hat{Q}_\pi(s,a|\boldsymbol\xi) = \sum_{t=0}^{H-1}\alpha^t c_t$, $M$개 평균 → $\hat{q}_\pi(s,a)$.
- $\mathcal{S}$에서 $N$개 상태만 표본 → 각 상태의 **simulation-based action** $\hat\pi^+(s)$ 산출.
- 전 상태로 확장하려면 정책 근사 필요 → **분류 문제**로 환원.

### 4.2 DCL (Algorithm 1) — §3.2
$n$번 반복(논문 $n=3$):
1. **State sampling**: 각 스레드가 길이 $L$ 워밍업(`SampleStartState`, Alg.2)으로 현실적 시작 상태 확보. 이후 상태는 $\hat\pi^+$ 따라 전이(실제 방문 분포 반영).
2. **Simulation-based action** (`Simulator`=SH+CRN, Alg.3): 각 샘플 상태의 최적 행동 추정 → $(s_k, \hat\pi^+(s_k))$를 데이터셋 $\mathcal{K}_i$에 추가.
3. **Classifier** (Alg.4): 신경망 분류기로 $\hat\pi^+(\cdot)$ 근사 → $\pi_{i+1}$. 다음 반복의 초기 정책.
- 스레드 $w$개 병렬, 각 $\lceil N/w \rceil$ 샘플.

### 4.3 Sequential Halving + CRN (Algorithm 3) — 효율의 핵심
- 상태당 총예산 $B_s = M|\mathcal{A}_s|$ 고정.
- **SH**: $\lceil\log_2|\mathcal{A}_s|\rceil$ 라운드. 각 라운드 후 하위 절반 행동 제거, 생존 행동에 예산 집중(best-arm identification, 하이퍼파라미터 없음).
- **CRN**: 같은 외생 시나리오를 경쟁 행동 모두에 공유 → $\mathrm{Var}[X] = \mathrm{Var}[\hat Q_a] + \mathrm{Var}[\hat Q_{a'}] - 2\,\mathrm{Cov}[\hat Q_a,\hat Q_{a'}]$에서 양의 공분산 항이 차이의 분산을 크게 줄임.
- **Stockpiling**: 원조 SH와 달리 라운드 종료 시 비용 합을 버리지 않고 누적 → 분산 추가 감소.

### 4.4 분류기 학습 (Algorithm 4) — Appendix C
- 입력 (상태→행동) 데이터셋, **cross-entropy loss**(불가행동 마스킹), Adam.
- 정책 $\pi_\theta(s) = \arg\max_a \mathrm{NN}_\theta(s)[a]$.
- 비용을 함수근사하지 **않음** — 신경망은 **정책 표현 전용**(노이즈 회피).

---

## 5. 실험 결과

하이퍼파라미터 전 실험 공통(robustness 강조): $H=40$, $M=1000$, $N=5000$, $L=100$, $n=3$. MLP 4층 {256,128,128,128}, Adam, batch 64. C++20, AMD EPYC 7H12 128스레드.

| 문제 | 벤치마크 | DCL 결과 |
|------|----------|----------|
| **Lost sales** (Zipkin/Xin testbed, $p\in\{4,9,19,39\}$, $\tau\le10$) | CBS, Myopic-2, A3C(3–6%) | 최적오차 ≤0.09%, A3C 대비 ≥10배 개선. 일부 $\tau=10$ 대형은 N=20000으로 해결 |
| **Perishable** (Haijema testbed 81개, FIFO/LIFO) | BSP-low-EW | 소형 평균 최적오차 0.03%, 대형 BSP 대비 -8.9%(BSP-low-EW -4.3%) |
| **Random lead time** (Stolyar, 지수/균등/파레토) | Generalized BSP (GBS) | 지수분포서 최적정책 거의 일치, 전 분포서 GBS 능가 |

- 계산시간: 인스턴스당 100–300초 (A3C는 튜닝 포함 수일).

### SH+CRN 효과 (§5.2, ablation = DCL₀ = uniform allocation + no CRN)
- DCL₀도 휴리스틱은 능가(외생 시나리오 다중 평가의 효과 입증). 단 lost sales 대형선 BSP에 가끔 짐.
- SH+CRN 추가 시 최적오차 **한 자릿수 이상 감소**(동일 $M$).
- 정답 분류율: 주이득은 **CRN**(분산 급감), SH는 CRN과 함께 최대 +10%p. DCL₀가 DCL 성능 따라잡으려면 시나리오 **100배 이상** 필요.

---

## 6. 세 재고문제 MDP-EI 정식화 (Appendix A) — 구현 직결

- **Lost sales**: 상태 $s\in\mathbb{R}^\tau$ = (on-hand, 파이프라인 $\tau-1$개). 전이 $f = ((s[1]-\xi)^+ + s[2], s[3],\dots,s[\tau], a)$. 비용 $C = h(s[1]-\xi)^+ + p(\xi-s[1])^+$. $\pi_0$=BSP(base-stock = 수요 $\tau$기간 newsvendor fractile $I_{max}$), $\mathcal{A}_s$는 inventory position ≤ $I_{max}$ 보장.
- **Perishable**: 상태 = 잔여수명별 재고($m_l$) + 파이프라인. 외생입력 $\xi=(\xi^{FIFO},\xi^{LIFO})$. 비용 = 폐기 $w\,n_{perish}$ + 보유 + 페널티. $I_{max}=m$ = $\tau+m_l$기간 newsvendor.
- **Random lead time**: 상태 $(s[1]$ on-hand$, s[2]$ 파이프라인 총량$, \mathbf{x}$ 각 주문 경과시간$)$. order-crossing 때문에 전체 주문이력 필요. 외생입력으로 차기 epoch 도착여부·간격 결정. inventory position·backorder 절단으로 유한화.

---

## 7. 본 repo 구현과의 대응 / 갭 (적용 노트)

현재 repo: `src/dcl_definitions.py`, `notebooks/dcl_Classifier.ipynb`, `notebooks/dcl_with_pth.ipynb`, `checkpoints/`.

| 논문 요소 | repo 대응 | 갭/할 일 |
|-----------|-----------|----------|
| Alg.1 DCL 반복 | Classifier 노트북 | $n$=정책반복 루프·세대별 평가 확인 |
| Alg.3 SH+CRN | ? | **CRN 공유·stockpiling 구현 여부 점검** — 미적용 시 정답 분류율 급락 (논문 핵심) |
| $\pi_0$=BSP, $\mathcal{A}_s$ 제약 | ? | base-stock 초기화·행동마스킹 검증 |
| cross-entropy 분류 | dcl_Classifier | 손실·마스킹 일치 확인 |
| 휴리스틱 벤치마크(CBS/BSP-low-EW/GBS) | ? | **벤치마크 재현 없으면 "휴리스틱 대비 개선" 주장 불가** (TODO 진단 항목) |

> **DCL이 휴리스틱을 못 이기는 현재 repo 증상의 1순위 용의자**: (a) CRN 미공유로 행동가치 비교 분산 폭발, (b) $M$/$H$ 부족, (c) 행동집합 $\mathcal{A}_s$ 미제약/초기 BSP 부재, (d) 정책반복 1회만 수행. 논문 §5.2가 정확히 이 인과를 입증함 — 진단 시 이 표를 체크리스트로.

---

## 8. ks_wiki Concepts 반영 (개념 정리 대상)

신규/보강 필요 개념: **MDP-EI**, **Approximate Policy Iteration**, **Classification-Based Policy Iteration(CBPI, 약어 정정)**, **Rollout Simulation(값 추정용)**, **Capped Base-Stock**, **Myopic Policy**, **BSP-low-EW**, **Generalized Base-Stock**, **Perishable Inventory**, **Random Lead Time / Order Crossing**, **Best-Arm Identification(SH 상위)**, **Cross-Entropy Loss**.
기존 정정: `DCL-CBPI.md`의 CBPI 약어·가공된 "BCL" 비교표.
