# Repo 코드 진단 — 논문(arXiv:2011.15122) §7 체크리스트 대비

> 진단일 2026-06-05. 대상: `src/dcl_definitions.py`, `notebooks/dcl_Classifier.ipynb`, `notebooks/dcl_with_pth.ipynb`.
> 결론: **"환경이 단순해서 개선이 안 됐다"가 아니다. 라벨(최적행동)을 오염시키는 실제 버그 + 측정 부재가 원인.**

---

## TL;DR — 심각도순

| # | 심각도 | 문제 | 위치 |
|---|--------|------|------|
| C1 | 🔴 CRITICAL | 시뮬레이터 CRN 시나리오의 **요일 미정렬** → 라벨 오염 | `dcl_definitions.py:361` |
| C2 | 🔴 CRITICAL | **벤치마크 평가 코드 자체가 없음** → "개선 없음" 미측정 | notebooks 전체 |
| C3 | 🔴 CRITICAL | M·N·H·L·n **과소 설정** → 라벨 품질 붕괴 | notebook 셀1·셀2 |
| H1 | 🟠 HIGH | **Aₛ 행동 미제약**(I_max 캡·마스킹 없음) → 예산 낭비 | `:371` |
| H2 | 🟠 HIGH | 분류 손실/추론에 **불가행동 마스킹 없음** | `:94`, `:64` |
| H3 | 🟠 HIGH | 비-정상(주간 계절) 수요 = 논문 외 확장, C1과 결합해 학습난이도↑ | `:170`, `:165` |
| M1 | 🟡 MED | SH 선택이 평균 아닌 **누적합**으로 비교(명명 오류) | `:408` |
| M2 | 🟡 MED | `tau=1` 시 action이 상태에 미반영 | `:195-198` |
| M3 | 🟡 MED | 노트북 import 이름 불일치(`_final`/`_cuda` vs 실제 파일) → 재현 불가 | notebooks |
| M4 | 🟡 LOW | 비용 스케일 비현실(h=1000 등), 디버깅 난이도 | notebook |

---

## CRITICAL

### C1. 시뮬레이터 CRN 시나리오의 요일 미정렬 (스모킹건)
`_run_simulation_steps()`:
```python
scenario = env.get_exogenous_scenario(length=H)   # line 361 — start_day_index 누락!
```
`get_exogenous_scenario(length, start_day_index=0)` 기본값이 **월요일**. 그런데 평가 대상 `state`는 임의 요일(요일 one-hot 보유). `rollout()`은 `state`의 요일을 따라 `get_next_state`로 전이하지만, **demand 시퀀스는 월요일 기준으로 생성**된다 → 요일↔수요 어긋남.

- 결과: 행동가치 $\hat q(s,a)$ 가 그 상태의 실제 요일 수요분포가 아닌 엉뚱한 분포로 계산됨.
- 그 $\arg\min$ 으로 뽑은 **simulation-based action(=분류기 라벨)이 체계적으로 틀림**.
- 주간 계절성(H3)이 있는 한 분류기는 잘못된 정책을 학습 → 휴리스틱을 못 이기는 직접 원인.
- **수정**: simulator/_run_simulation_steps에 현재 state의 요일을 넘겨 `get_exogenous_scenario(length=H, start_day_index=current_day)` 로 생성. (또는 H3을 제거해 정상 수요로 전환.)

### C2. 벤치마크 평가 코드 부재
노트북은 `train()` 후 `best_policy.get_action(example_state)` 하나 출력할 뿐. **DCL·BSP·휴리스틱의 평균비용/기간을 측정하는 시뮬레이션 평가 루프가 전무.**
- 논문 평가: 1000 run × 5000 period, warm-up 100, 95% CI 반폭 <1%.
- "휴리스틱 대비 개선 없음"이라는 판단의 **근거 자체가 측정되지 않았다.**
- **수정**: `evaluate_policy(env, policy, runs, horizon, warmup)` 작성 → DCL vs BSP vs (CBS 등) 평균비용 비교 표 생성. 이게 있어야 모든 후속 진단이 정량화됨.

### C3. 시뮬레이션 예산 과소
| 파라미터 | 노트북 | 논문 | 영향 |
|---|---|---|---|
| M (시나리오/행동) | **10** | 1000 | 라벨 분산 폭발 |
| N (샘플 수) | 300 | 5000 | 상태공간 미커버 |
| H (rollout 깊이) | 20 | 40 | 절단 편향 |
| L (워밍업) | 10 | 100 | 시작상태 비현실 |
| n (정책반복) | 2 | 3 | 미수렴 |

주석 "오래걸려서 낮춤"이 성능을 직접 죽인다. 논문 §5.2: CRN/SH 없으면 동일 정확도에 100배 시나리오 필요 — M=10은 그 반대 방향. CRN 버그(C1)까지 겹쳐 라벨 신뢰도 바닥.

---

## HIGH

### H1. Aₛ 행동 미제약
```python
A_s = list(range(env.num_actions))   # line 371 — 0..max_action 전부 항상 후보
```
`num_actions = max_action+1 = 301` (mean demand 150인데 max_action=300). 논문은 inventory position ≤ $I_{max}$ 로 $\mathcal{A}_s$ 를 상태별 제약. 301개 행동 전부 SH 평가 = 예산을 무의미한 대량주문에 낭비 + 라운드수 $\lceil\log_2 301\rceil=9$.
- **수정**: 상태별 feasible 행동집합(주문 후 inventory position ≤ $I_{max}$)으로 `A_s` 구성, BSP base-stock을 newsvendor fractile로 설정.

### H2. 불가행동 마스킹 없음
`train_policy`의 `nn.CrossEntropyLoss()` (line 94)와 `get_action`의 `argmax`(line 64) 모두 전 행동 대상. 논문은 $a\notin\mathcal{A}_s$ 를 softmax 분모에서 마스킹. H1과 함께 정책이 불가/과대 주문을 출력 가능.

### H3. 비-정상(주간 계절) 수요 — 논문 외 확장
```python
self.daily_mean_demands = np.maximum(1, np.random.normal(mean, std, size=7))  # line 170
self.state_size = self.tau + 7   # 요일 one-hot
```
논문은 i.i.d. **정상** 수요. 본 repo는 요일별 평균이 다른 계절 수요로 확장하고 state에 요일 7차원을 더했다. → 사용자가 "환경이 단순"이라 본 것과 정반대로 **더 복잡**. 확장 자체는 가능하나 (a) C1 버그를 유발했고 (b) 상태공간을 키워 N=300으로는 학습 불가. **결정 필요**: 논문 재현(정상 수요로 단순화) vs 계절 확장 유지(그러면 C1 필수 수정 + N·M 대폭 증대).

---

## MEDIUM / LOW

- **M1** `:408` `avg_costs = {a: accumulated_costs[a] ...}` — 이름은 평균인데 실제 누적합. 현 라운드 생존 행동은 동일 시나리오 수라 랭킹엔 무해하나, stockpiling 의도와 혼동·잠재 버그. 평균(`/T`)으로 명시 권장.
- **M2** `:195-198` lost sales `tau=1`이면 action이 다음 상태에 안 들어감(논문 τ>1 가정이라 실용상 OK, 방어코드 권장).
- **M3** 노트북 `from dcl_definitions_final import ...` / `dcl_definitions_cuda` — 실제 파일은 `dcl_definitions.py`. 그대로면 ImportError. 재현성 차단.
- **M4** h=1000·p=5000·demand 150 등 스케일 비현실(논문 h=1). 상대비교엔 무관하나 디버깅·해석 저해.

---

## 권장 수정 순서

1. **C2 평가 하니스 먼저** — 정량 측정 없이는 어떤 수정도 검증 불가. DCL vs BSP 평균비용 루프.
2. **H3 결정** — 정상 수요로 논문 재현 모드 추가(요일 제거) 권장. 디버깅·재현이 쉬워지고 C1 자동 해소.
3. **C1 수정** — (H3 계절 유지 시 필수) CRN 시나리오 요일 정렬.
4. **C3 예산 복원** — M·N·H·L·n 을 논문값 근처로. 정상 수요 모드면 부담↓.
5. **H1·H2 Aₛ 제약 + 마스킹**.
6. M1·M2·M3·M4 정리.
7. 재평가 → lost sales·perishable에서 BSP·CBS·BSP-low-EW 대비 비용 표 기록 → README·포트폴리오 수치 갱신.

> 핵심 메시지: 코드 구현이 "정확한데 환경이 단순"한 게 아니라, **라벨 생성 경로(C1)와 측정 부재(C2), 예산 과소(C3)** 가 겹쳐 DCL이 제 성능을 낼 수 없는 상태. 논문의 SH+CRN 효율 이점이 M=10·CRN 버그로 무력화돼 있다.

---

## Plan A 실행 결과 분석 (2026-06-05, `src/dcl_paper.py`)

정상수요 재현판으로 lost sales(Poisson mean=5, τ=3, h=1, p=9) 실험. I_max=20, m=8.

### v1 결과 (N=800, M=60, H=30)
```
BSP* 6.79 | CBS* 6.60
DCL gen1=gen2=gen3 = 7.94  (CBS* 대비 +20%)  ← 셋 다 동일값
```

### 추가 발견 (진단 스크립트 `scripts/diag.py`, `diag2.py`)
- **B1 (보고 버그)**: `DCLTrainer.train` 의 `policies.append(clf)` 가 **동일 clf 객체를 n번** 담아 세대 비교 불가 → "gen 동일"은 학습 정체가 아니라 객체 중복. **수정**: 반복마다 새 Classifier 생성(논문 π_{i+1}=Classifier(K_i)).
- **라벨은 붕괴 아님**: 라벨 분포 {3:34,4:73,5:112,6:42,...} 정상(5 중심). 분류기 행동맵도 상태별 변화.
- **진짜 원인 = 예산 과소로 인한 라벨 편향(논문 §5.2 그대로)**: 참 q값(독립 rollout 4000개 ≈ exact)에서 저재고 상태의 q 지형이 **평평**(state[0,0,0] H=30: q(5)=351.7, q(6)=351.5, q(7)=351.3, q(8)=353.5). BSP 연속정책이 알아서 리필해 초기행동 차이를 가린다. 참 argmin=7 인데 simulator(M=500)는 6 선택(약간 under-order). 이게 분류기로 일반화돼 **저재고서 리필 부족 → 품절 → 비용↑.** 중간재고[5,5,4]선 simulator=BSP 일치(문제 없음).
- 결론: **알고리즘/CRN/SH 로직은 정상.** M=60·H=30·N=800 이 라벨을 충분히 날카롭게 못 만들어 DCL이 BSP 미만. 논문 예산(M=1000,H=40,N=5000)이 필요한 이유를 실측으로 재확인.

### v2 (B1 수정, M=400·H=40·N=800): 여전히 7.93 (+20%)
M 60→400 올려도 동일 → 예산 가설 기각. 7.93 ≈ base-stock S=20 비용(diag3: S=20→8.06, S=23→6.82). 상수주문은 전부 끔찍(order5→43) → 상수붕괴도 아님.

### B2 (진짜 단일 원인): I_max 캡 과소
- `I_max = poisson.ppf(0.9, mean*τ=15) = 20`. `feasible_actions` 가 inventory position ≤ I_max=20 강제.
- 그러나 **최적 base-stock = 23 > 20** → DCL 이 최적 재고수준 도달을 구조적으로 금지당함 → BSP* 못 이김.
- lost sales 는 backorder 보다 안전재고가 더 필요해 τ기간 newsvendor 가 최적을 과소추정. **수정: (τ+1)기간 → I_max=26.**

### v3 (B2 수정, M=250·H=40·N=800): ✅ DCL 이 휴리스틱 능가
```
BSP*(23) 6.8161 | CBS*(24,6) 6.6155
DCL gen1 6.5668 (vs CBS* -0.74% | vs BSP* -3.66%)
```
모자란 예산에도 BSP -3.66%, CBS -0.74% 능가(CI ±0.03, 유의). **I_max 캡이 단일 원인 확정. 알고리즘 정상.**

### 최종 결론
- "환경 단순" 아님. 원인은 (기존 노트북) C1 요일버그+M=10, (재현판) **B2 I_max 캡**.
- 다음: full(M=1000,N=5000) 1회로 논문 수치(최적해 대비 ≤0.2%) 재현, perishable 동일 점검, README·포폴 수치 갱신.
