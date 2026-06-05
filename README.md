# 📦 Deep Controlled Learning (DCL) for Inventory Control

> **확률적 리드타임·부패성(Perishable) 재고 시스템 최적화 — 시뮬레이션 기반 Deep Controlled Learning**

![Role](https://img.shields.io/badge/Role-팀장(Team_Lead)-blue)
![Result](https://img.shields.io/badge/최적해_오차-0.01%25-success)
![Baseline](https://img.shields.io/badge/Capped_Base--Stock_대비--1.1%25-success)
![Efficiency](https://img.shields.io/badge/CRN_시뮬효율->10배-orange)
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)

## 💡 프로젝트 개요 (Overview)

본 프로젝트는 **확률적 리드타임(Stochastic Lead Time)과 부패성 재고(Perishable Inventory)** 환경에서 범용 강화학습(Policy Gradient)이 겪는 **수렴 불안정·낮은 샘플 효율** 문제를 극복하기 위한 솔루션입니다.

일반적인 RL 대신 **시뮬레이션 + 분류기(Classifier)** 조합인 **Deep Controlled Learning (DCL)** 패러다임을 채택하여, 극도로 복잡한 재고 운영 문제의 최적 정책을 효율적으로 학습합니다.

> 산업경영 인공지능 모델링 (성결대 2025-2) 팀 프로젝트 / **역할: 팀장 (전체 알고리즘 설계 주도)**
> 참고 논문: *Deep Controlled Learning for Inventory Control* (Lecture Note 1) — https://arxiv.org/abs/2011.15122

## 🎯 핵심 문제 (Problem)

| 항목 | 내용 |
|------|------|
| 환경 | 확률적 리드타임 + 부패성 재고 (Lost Sales / Perishable) |
| 한계 | 범용 RL은 고분산 환경에서 Policy Gradient 수렴 불안정, 탐색 공간 과대 |
| 접근 | 시뮬레이션으로 최적 행동을 추정하는 **Classification-Based Policy Iteration (CBPI)** |

## 🧠 방법론 — DCL 알고리즘 (Methodology)

![DCL Algorithm](./docs/images/dcl_algorithm_code.png)

핵심 기술 3가지:

1. **Sequential Halving (SH)** — 매 라운드 하위 50% 후보 행동 제거로 최적 후보를 효율적으로 좁힙니다(best-arm identification).
2. **CRN (Common Random Numbers)** — 동일 랜덤 시나리오를 후보 행동에 공유하여 비교 분산을 최소화합니다. ablation에서 **정답 분류율을 좌우하는 핵심 요소**로 확인되었습니다.
3. **CBPI (Classification-Based Policy Iteration)** — 시뮬레이션으로 현재 상태의 최적 행동을 추정하고, Classifier가 (상태 → 행동) 정책을 학습합니다.

### 알고리즘 구조

![Simulator Architecture](./docs/images/simulator_architecture.png)

```
DCLTrainer.train (Algorithm 1)
    └── n번 반복
        ├── SampleStartState (Algorithm 2)   ← L길이 워밍업 샘플 수집
        ├── Simulator (Algorithm 3)           ← SH 라운드 반복 + CRN
        │   ├── CRN으로 시나리오 생성
        │   ├── 평균비용 상위 절반 생존
        │   └── 최종 1개 (s, a*) 기록
        └── Classifier 정책 학습 (Algorithm 4) ← π 업데이트
```

## 📊 결과 (Results)

> 아래 수치는 **정상 i.i.d. 수요 재현 구현(`src/dcl_paper.py`)** 으로 측정한 값입니다. 평균비용은 독립 400 run × 2,000 기간 시뮬레이션으로 추정했고(95% CI 반폭 ≤ 0.02), Lost Sales는 평균비용 MDP를 **value iteration으로 정확히 풀어 최적해 v\*** 를 구한 뒤 optimality gap을 계산했습니다.

### Lost Sales (Poisson, τ=3, p=9, h=1) — 최적해 대비 gap

| 정책 | 평균비용 | Optimality Gap |
|------|---------|----------------|
| **최적해 v\*** (value iteration) | 6.531 | — |
| **DCL** (best generation) | 6.532 | **+0.01%** |
| Capped Base-Stock (CBS\*) | 6.602 | +1.08% |
| Base-Stock (BSP\*) | 6.802 | +4.15% |

→ DCL이 최적해를 **0.01% 오차**로 재현하며, 최적화된 휴리스틱(CBS·BSP)을 모두 능가합니다.

### Perishable (FIFO, lifetime 3) — 휴리스틱 대비

| 정책 | 평균비용 | 비고 |
|------|---------|------|
| **DCL** (best generation) | 5.478 | Base-Stock 대비 **−1.55%** |
| Base-Stock (BSP\*) | 5.564 | — |

### SH + CRN ablation — 정답 분류율(정확 one-step improvement 기준, %)

| 전략 | M=200 | M=2000 |
|------|-------|--------|
| **SH + CRN (DCL)** | 79.6 | **93.1** |
| Uniform + CRN | 69.4 | 91.6 |
| SH − CRN | 27.6 | 40.0 |
| Uniform − CRN (DCL₀) | 25.8 | 34.6 |

→ **CRN이 정확도를 좌우하는 주효과**(제거 시 절반 이하로 붕괴)이며, SH가 그 위에 최대 약 +10%p를 더합니다. DCL이 M=200에서 내는 정확도를 DCL₀는 M=2,000에서도 도달하지 못해 **시뮬레이션 효율 10배 이상**의 이점이 확인됩니다(논문은 약 2 orders of magnitude 보고).

> 📂 재현 스크립트·로그: `scripts/run_full.py`(최적해 gap), `scripts/run_perishable.py`, `scripts/run_ablation.py` / 결과 `knowledge/results_*.txt`.

### 🔍 디버깅 사례 — 휴리스틱을 못 이기던 원인 규명

초기 노트북 구현은 휴리스틱 대비 비용 개선이 없었습니다. 원논문을 정독하며 재구현·진단한 결과, 원인은 환경의 단순함이 아니라 **feasibility 제약의 버그**였습니다: 행동집합 제약 상한 `I_max`를 τ기간 newsvendor 분위수로 잡아 **최적 base-stock(=23)보다 낮은 값(=20)** 으로 설정해, 정책이 최적 재고수준에 도달하는 것을 구조적으로 막고 있었습니다. (τ+1)기간 기준으로 교정하자 DCL이 즉시 휴리스틱을 능가했습니다. 전체 진단 과정은 `knowledge/diagnosis_repo_vs_paper.md`에 정리했습니다.

## 📁 레포지토리 구조 (Repository Structure)

* `src/dcl_paper.py`: **논문 충실 재현 구현(권장)** — 정상 i.i.d. 수요 Lost Sales / Perishable 환경, Aₛ 제약 + 분류기 행동 마스킹, SH+CRN(stockpiling·평균비교), 평가 하니스, 휴리스틱(BSP·CBS) 최적화 일체
* `src/optimal.py`: 평균비용 MDP **정확 최적해**(relative value iteration) + 고정정책 평가/one-step improvement — optimality gap·ablation 정답기준 산출
* `scripts/`: 실험 러너 — `run_full.py`(최적해 gap), `run_perishable.py`, `run_ablation.py`(SH/CRN), 진단 `diag*.py`
* `knowledge/`: 논문 요약, 진단 보고서(`diagnosis_repo_vs_paper.md`), 실험 결과 로그(`results_*.txt`)
* `src/dcl_definitions.py`: 초기 노트북용 구현(요일 계절수요 확장판) — 진단을 통해 재구현으로 대체
* `notebooks/`: `dcl_Classifier.ipynb`(학습), `dcl_with_pth.ipynb`(추론). 주피터 병렬 제약으로 정의/학습 분리
* `docs/`: 발표 자료(PDF), 알고리즘·시뮬레이터 다이어그램

## 🚀 시작하기 (How to Run)

```bash
# 1. 의존성 설치
pip install torch numpy scipy

# 2. 재현 구현으로 전체 실험 (권장)
python scripts/run_full.py        # Lost Sales: 최적해 대비 optimality gap
python scripts/run_perishable.py  # Perishable: 휴리스틱 대비
python scripts/run_ablation.py    # SH/CRN 정답 분류율 ablation

# 3. (초기 버전) Jupyter 노트북
#    notebooks/dcl_Classifier.ipynb (학습) / dcl_with_pth.ipynb (추론)
```

> 시뮬레이션이 병목이므로 CPU 멀티프로세싱(`w`)이 핵심이며 GPU 이점은 없습니다(원논문도 CPU 사용). DCL 학습부는 PyTorch가 필요하나, 환경·시뮬레이터·평가·휴리스틱은 NumPy/SciPy만으로 동작합니다.

## 🔗 관련 자료

* 핵심 알고리즘 개념: CBPI / Sequential Halving / CRN
* End-to-End SCM AI 파이프라인의 **Phase 3 (Inventory / 재고운영)** 단계
