# DCL Inventory — 작업 핸드오프 (2026-06-05)

> 이 문서는 다른 세션 / 미래의 나를 위한 **작업 일지 + 코드 설명서**입니다.
> 포트폴리오용 요약은 `README.md`, 진단 상세는 `knowledge/diagnosis_repo_vs_paper.md`, 논문 요약은 `knowledge/summary_dcl_inventory_control.md` 참조.

---

## 0. 한 줄 요약

원논문(arXiv:2011.15122) **Deep Controlled Learning** 재고관리 구현이 "휴리스틱을 못 이긴다"는 문제를 진단한 결과, 원인은 환경의 단순함이 아니라 **feasibility 캡 버그(I_max)** 였다. 정상 i.i.d. 수요로 재구현하고 버그를 고치자 DCL이 **최적해 오차 0.01%** 로 논문 수치를 재현하고 휴리스틱(BSP·CBS)을 능가했다. SH/CRN ablation도 논문 정성 결과를 재현했다.

---

## 1. 배경 / 목적

- 프로젝트: 산업경영 인공지능 모델링(성결대 2025-2) 팀 프로젝트, 역할 = 팀장.
- 원논문: Temizöz et al., *Deep Controlled Learning for Inventory Control*, arXiv:2011.15122.
- 출발점 문제: 기존 노트북 구현(`src/dcl_definitions.py` + `notebooks/`)이 base-stock 등 휴리스틱 대비 비용 개선이 없었음. "환경이 너무 단순해서"라고 추정했으나 실제로는 코드 문제였음.

### DCL 알고리즘 핵심 (3줄)
- 재고문제는 **MDP with Exogenous Inputs (MDP-EI)** — 수요·리드타임이 외생 확률원.
- DCL = **근사 정책 반복을 분류 문제로 환원(CBPI = Classification-Based Policy Iteration)**. 시뮬레이션으로 각 상태의 최적 행동(label)을 추정 → 신경망 분류기가 (상태→행동) 정책 학습 → 반복.
- 시뮬레이션 효율의 핵심 = **Sequential Halving(SH) + Common Random Numbers(CRN)**.

---

## 2. 진단 여정 (가설이 두 번 바뀜)

| 단계 | 가설 | 결과 |
|------|------|------|
| 노트북 정독 | C1 시뮬레이터 CRN 시나리오 요일 미정렬, C2 평가코드 부재, C3 예산 과소(M=10), H1 Aₛ 미제약, H3 비정상 계절수요 | 정상수요로 재구현 결정 |
| v1 (M=60) | — | DCL 7.94 (+20% 나쁨), gen 3개 동일 |
| **B1** | 보고 버그: `policies.append(clf)` 동일 객체 n번 | 수정(반복마다 새 분류기). 여전히 7.93 |
| v2 (M=400) | 예산 과소? | 7.93 그대로 → **기각** |
| diag3 | 상수주문 붕괴? | 상수주문 전부 끔찍(order5→43) → **기각**. 7.93 ≈ base-stock 20 비용 |
| **B2 (진짜 원인)** | **I_max 캡 과소** | DCL 7.94 ≈ base-stock 20 = I_max. 최적 base-stock 23 > 20 → feasibility가 최적정책 차단 |
| v3 (B2 수정) | — | DCL **6.567**, CBS −0.74%, BSP −3.66% ✅ |
| full | 강한 예산 + 정확 최적해 | DCL best gap **+0.01%** |

### 버그 정리
- **B2 (핵심)**: `LostSalesEnv`의 `I_max`(inventory position 상한 캡)를 **τ기간 newsvendor 분위수(=20)** 로 잡았는데, lost sales 최적 base-stock(=23)이 그보다 높음 → 정책이 최적 재고수준 도달을 구조적으로 금지당함. 수정: **(τ+1)기간 newsvendor(=26)**. lost sales는 backorder보다 안전재고가 더 필요해 τ기간 추정이 과소였다.
- **B1**: `DCLTrainer.train`이 같은 분류기 객체를 매 반복 append → 세대 비교 불가. 수정: 반복마다 새 `Classifier` 생성(논문 π_{i+1}=Classifier(K_i)).
- C1/C2/C3/H1/H2/H3: 기존 노트북의 문제들(요일 계절수요 확장이 유발한 CRN 미정렬, 평가코드 부재, M=10 과소 등). 정상수요 재구현으로 해소.

---

## 3. 완성된 코드 설명

### `src/dcl_paper.py` — 논문 충실 재현 구현 (메인, ~370줄)
torch는 분류기/학습에만 필요하도록 **지연 import** → 환경·시뮬레이터·평가·휴리스틱은 NumPy/SciPy만으로 동작.

| 구성 | 설명 |
|------|------|
| `Demand` | 정상 i.i.d. 수요(poisson/geometric), `sample`·`ppf`(newsvendor 분위수) |
| `LostSalesEnv` | 상태=[on-hand, pipeline]. 전이 `s'=((s0-d)^+ + s1, …, a)`, 비용 `h(s0-d)^+ + p(d-s0)^+`. **`feasible_actions`: inventory position ≤ I_max 제약**. `I_max`=(τ+1)기간 newsvendor(**B2 수정 지점**), `m`=1기간 newsvendor |
| `PerishableEnv` | 상태=잔여수명별 재고+pipeline. FIFO/LIFO 판매(`_sell`), 폐기·보유·품절 비용. I_max는 (τ+ml)기간 기준 |
| `BaseStockPolicy` / `CappedBaseStockPolicy` | 휴리스틱. `get_action(state, env)` |
| `Classifier` | 4층 MLP(256,128,128,128). `get_action`에서 **불가행동 마스킹**(-inf), `train_policy`(CE loss, Adam, early stopping) |
| `sample_start_state` | Alg 2. L길이 워밍업 |
| `rollout` | 절단 궤적 누적비용(t=0 initial action, 이후 π) |
| `simulator` | **Alg 3 = SH + CRN**. 라운드별 하위 절반 제거, 시나리오 공유(CRN), 누적합 stockpiling, **평균으로 비교** |
| `collect_samples` | 상태 표본 + 각 상태의 simulation-based action(=label) 수집 |
| `DCLTrainer.train` | Alg 1. n회 반복, w워커 멀티프로세싱, **반복마다 새 분류기(B1 수정)** |
| `evaluate_policy` | 독립 run × horizon 평균비용 + 95% CI |
| `optimize_base_stock` / `optimize_capped_base_stock` | 그리드 서치로 BSP*/CBS* 최적 파라미터 |

### `src/optimal.py` — 정확 최적해 + 정답기준
- `solve_lost_sales_optimal`: 평균비용 MDP를 **relative value iteration**으로 정확히 풀어 v*(최적 평균비용)·최적정책 산출. optimality gap 계산용.
- `policy_evaluate_lost_sales`: 고정 BSP(S) 정책 평가 + **정확한 one-step improvement π⁺(s)** = ablation 정답기준.
- 상태공간 = inventory position 합 ≤ I_max 인 정수벡터(τ작은 소형 인스턴스 전용).

### `src/dcl_definitions.py` — 초기(노트북용) 구현
요일 계절수요 확장판. B2/C1 등 버그 보유. **진단 대상**이었고 `dcl_paper.py`로 대체됨. 보존만.

### `scripts/` — 실험 러너
| 파일 | 용도 |
|------|------|
| `run_full.py` | Lost Sales: 최적해 v* 기준 optimality gap (DCL/BSP/CBS) — **메인 결과** |
| `run_perishable.py` | Perishable: DCL vs BSP* |
| `run_ablation.py` | SH/CRN ablation: 4전략 정답분류율 vs M |
| `run_v2.py`/`run_v3.py` | 진단 중간 검증(B1 후/B2 후) |
| `run_lost_sales.py` | v1(초기) |
| `diag.py`/`diag2.py`/`diag3.py`/`diag_perishable.py` | 라벨분포·참 q값·상수주문/base-stock 스윕·perishable I_max 점검 |

### `knowledge/` — 분석·결과
- `summary_dcl_inventory_control.md`: 논문 요약(알고리즘·실험·repo 갭 체크리스트)
- `diagnosis_repo_vs_paper.md`: 진단 전체 보고서(버그 C1~H3, B1, B2, 실행결과 분석)
- `results_*.txt`: 각 실험 결과 / `run_*_log.txt`: 콘솔 로그

---

## 4. 실험 결과 (전부 실측·검증)

### Lost Sales (Poisson, τ=3, p=9, h=1) — optimality gap
| 정책 | 평균비용 | gap |
|------|---------|-----|
| 최적해 v* (value iteration) | 6.531 | — |
| **DCL best** | 6.532 | **+0.01%** |
| CBS* (S=24, cap=6) | 6.602 | +1.08% |
| BSP* (S=23) | 6.802 | +4.15% |

(DCL: N=2000, M=500, H=40, n=3, w=12. 논문 목표 ≤0.2% 달성.)

### Perishable (FIFO, lifetime 3, τ=1) — 휴리스틱 대비
| 정책 | 평균비용 |
|------|---------|
| **DCL gen2** | 5.478 (BSP* 대비 **−1.55%**) |
| BSP* (S=11) | 5.564 |

(여기선 정책반복이 보정 역할: gen1은 초기 BSP(I_max) 과잉재고라 나빴고 gen2가 개선.)

### SH/CRN ablation — 정답분류율(%) vs M
| 전략 | M=200 | M=2000 |
|------|-------|--------|
| **SH+CRN (DCL)** | 79.6 | **93.1** |
| Uniform+CRN | 69.4 | 91.6 |
| SH−CRN | 27.6 | 40.0 |
| Uniform−CRN (DCL₀) | 25.8 | 34.6 |

→ **CRN이 주효과**(제거 시 절반 이하), SH가 +10%p 가산, DCL₀ 대비 효율 **>10배**(논문 ~100배는 M~2만 필요, 하한만 측정).

---

## 5. 실행법

```bash
# Python 3.12 + torch(cpu) + numpy + scipy 필요 (py3.14는 torch 휠 없음)
# 이 PC: py -3.12 사용. 시뮬이 병목이라 CPU 멀티프로세싱(w)이 핵심, GPU 이점 없음.

py -3.12 scripts/run_full.py        # 최적해 gap (~2.3시간, w=12)
py -3.12 scripts/run_perishable.py  # perishable
py -3.12 scripts/run_ablation.py    # SH/CRN ablation (~44분)
py -3.12 src/optimal.py             # 최적해 v* 단독 계산 (수초)
```
주의: Windows 콘솔 cp949라 비-ASCII(≈ 등) print 시 크래시 → 스크립트들은 `sys.stdout.reconfigure(utf-8)` 처리됨.

---

## 6. 남은 작업 (전부 선택)

- [ ] Plan B: 비정상 주간 계절수요 버전 별도 구현(C1 수정 + 계절판 BSP 재정의)
- [ ] antigravity `Desktop/Between Laptop/deep_controlled_learning` (paper2code 산출물)과 병합 — 그쪽은 알고리즘 코드가 깔끔하나 환경이 toy mock(리드타임 없음)·π₀가 NN·α=0.99. 내 진짜 env/휴리스틱/평가와 합치면 좋음.
- [ ] Perishable 정확 최적해 + BSP-low-EW 휴리스틱 비교(상태공간 커서 후속)
- [ ] 논문 정확 "~100배" 핀포인트(ablation을 M~2만까지, 수 시간)
- [ ] Docker 배포(전체 프로젝트 공통 todo, 맨 마지막)

---

## 7. 파일 위치 맵

```
Github/Deep-Controlled-Learning-for-Inventory-Control/
├── README.md                          # 포폴용 (실측치 반영 완료)
├── HANDOFF.md                         # 이 문서
├── src/
│   ├── dcl_paper.py                   # ★ 메인 재현 구현
│   ├── optimal.py                     # ★ 정확 최적해 + π⁺
│   └── dcl_definitions.py             # 초기 노트북용(대체됨)
├── scripts/                           # 실험 러너 + 진단
├── knowledge/                         # 논문요약·진단보고·결과로그
├── notebooks/                         # 초기 학습/추론 노트북
└── docs/                              # 발표자료·다이어그램

관련 ks_wiki:
- Wiki/Projects/Deep-Controlled-Learning-Inventory.md  (실측치 반영)
- Wiki/Concepts/DCL-CBPI.md 외 12종 (MDP-EI·API·CBPI·Rollout·SH·CRN 등)
```
