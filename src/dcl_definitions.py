# dcl_definitions.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from abc import ABC, abstractmethod
import math
import scipy.stats as stats # Perishable Env에서 사용

# ===================================================================
# 모델 및 정책 클래스
# ===================================================================

# <--- GPU 수정 1: 사용할 장치를 전역적으로 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# DCL 학습의 시작점이 되는 초기 정책 (예: Base-Stock Policy)
class BaseStockPolicy:
    """
    초기 정책 pi_0로 사용될 간단한 재고 보충 정책입니다.
    미리 정의된 재고 수준(base_stock_level)까지 주문합니다.
    """
    def __init__(self, base_stock_level):
        self.level = base_stock_level
    def get_action(self, state):
        # 파이프라인 재고를 포함한 현재 재고 포지션 계산
        inventory_position = np.sum(state)
        # 목표 수준까지 도달하기 위해 필요한 양을 계산
        action = max(0, self.level - inventory_position)
        return int(round(action))

# 알고리즘 4: 신경망 분류기 (Classifier) 정의
class Classifier(nn.Module):
    """
    상태(state)를 입력받아 최적 행동(action)을 예측하는 신경망 모델입니다.
    DCL의 정책 근사(Policy Approximation)를 담당합니다.
    """
    def __init__(self, input_size, num_actions):
        super(Classifier, self).__init__()
        # 논문에 제시된 신경망 구조 예시를 따름 (4-layer MLP)
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),  # ReLU 활성화 함수
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions) # 최종 출력은 각 행동에 대한 점수(logit)
        )
    def forward(self, state):
        # 상태 텐서를 입력받아 각 행동에 대한 로짓을 반환
        return self.network(state)
    def get_action(self, state):
        """학습된 모델을 정책으로 사용할 때 호출되는 함수"""
        if isinstance(state, np.ndarray):
            # <--- GPU 수정 2: 상태 데이터를 device로 보냄
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.forward(state)
            action = torch.argmax(logits, dim=1).item()
        return action
    
    def train_policy(self, dataset_K, epochs=50, batch_size=64, learning_rate=1e-3):
        """
        알고리즘 4(딥러닝)의 학습 과정을 구현합니다.
        (상태, 최적 행동) 데이터셋 K를 이용해 신경망을 학습시킵니다.
        """
        # 1. 데이터 준비: 데이터셋을 텐서로 변환하고 DataLoader 생성
        states, actions = zip(*dataset_K)
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(np.array(actions))
        
        if len(states_tensor) < 5:
            print("데이터셋이 너무 작아 학습을 건너뜁니다.")
            return self
            
        # 훈련/검증 데이터셋 분할 (예: 80/20)
        dataset_size = len(states_tensor)
        val_size = max(1, int(0.2 * dataset_size))
        train_size = dataset_size - val_size
        
        full_dataset = TensorDataset(states_tensor, actions_tensor)
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 2. 옵티마이저 및 손실 함수 정의
        optimizer = optim.Adam(self.parameters(), lr=learning_rate) # 옵티마이저로 가장 보편적인 아담 사용
        loss_fn = nn.CrossEntropyLoss() # 분류 문제의 표준 손실 함수

        best_val_loss, patience_counter, patience = float('inf'), 0, 5  # 조기 종료(Early Stopping)를 위한 대기 횟수
        print("신경망 학습 시작...")
        # 3. 학습 루프
        for epoch in range(epochs):
            self.train() # 모델을 훈련 모드로 설정
            for batch_states, batch_actions in train_loader:
                # <--- GPU 수정 3: 훈련 데이터 배치를 device로 보냄
                batch_states = batch_states.to(device)
                batch_actions = batch_actions.to(device)
                # 순전파
                outputs = self.forward(batch_states)
                loss = loss_fn(outputs, batch_actions)
                
                # 역전파 및 파라미터 업데이트
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 4. 검증 및 조기 종료
            self.eval() # 모델을 평가 모드로 설정
            current_val_loss = 0
            with torch.no_grad():
                for batch_states, batch_actions in val_loader:
                    # <--- GPU 수정 4: 검증 데이터 배치를 device로 보냄
                    batch_states = batch_states.to(device)
                    batch_actions = batch_actions.to(device)
                    outputs = self.forward(batch_states)
                    loss = loss_fn(outputs, batch_actions)
                    current_val_loss += loss.item()
            
            avg_val_loss = current_val_loss / len(val_loader) if val_loader else 0
            print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss, patience_counter = avg_val_loss, 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print("조기 종료(Early stopping) 발동")
                break
        print("신경망 학습 완료.")
        return self # 학습된 모델 반환
    
# ===================================================================
# 환경 클래스
# ===================================================================

# Abstract Base Class= 설계도, 계약서 = 이 클래스를 상속받는 모든 클래스는 반드시 특정 구조를 가져야한다.
# state, action, demand, cost의 형태를 고정시킴.
class BaseInventoryEnv(ABC):
    @abstractmethod
    def get_initial_state(self, start_day_index=0): # 입력값은 없음. 하지만 필수 구현 내용으로써 시뮬레이션이 시작될 때의 초기 상태를 반환해야함.
        pass
    @abstractmethod
    def get_next_state(self, state, action, demand): # 고정된 형태: State,Action, Demand를 입렵으로 받음. 그리고 현재 State에서 action하고 demand를 겪었을 때 다음 시점의 상태를 계산(상태 전이 함수 f 혹은 그 역할)하여 반환해야 함
        pass
    @abstractmethod
    def get_cost(self, state, action, demand): # state, action, demand를 입력으로 받습니다.필수 구현 내용: 현재 상태에서 특정 행동과 수요로 인해 발생하는 비용(재고 유지비, 판매 손실비 등)을 계산하여 반환해야 합니다. (비용 함수 C의 역할)
        pass
    @abstractmethod
    def get_exogenous_scenario(self, length): # 고정된 형태: 시나리오의 길이 length를 입력으로 받습니다. 필수 구현 내용: 시뮬레이션에 사용될 무작위 외부 요인(예: 수요)의 시나리오를 지정된 길이만큼 생성하여 반환해야 합니다. 이때 Simulator에서는 length = H, Sample Start State에선 웜업 기간 L,collect_samples_for_thread 함수에선 length는 해당 스레드가 수집할 **샘플의 개수(약 N/w)**
        pass

# 1 Lost Sales 문제 환경
class LostSalesEnv(BaseInventoryEnv):
    def __init__(self, lead_time, holding_cost, penalty_cost, mean_demand, demand_std_dev, max_action):
        self.tau = lead_time
        self.h, self.p = holding_cost, penalty_cost
        self.num_actions = max_action + 1
        self.state_size = self.tau + 7 # <<< 상태 크기 = 리드타임 + 요일(7)
        # 인스턴스 변수로 저장
        self.mean_demand = mean_demand
        self.demand_std_dev = demand_std_dev
        
        self.daily_mean_demands = np.maximum(1, np.random.normal(loc=mean_demand, scale=demand_std_dev, size=7))
        print("="*30 + "\nLost Sales: 자동 생성된 요일별 평균 수요(포아송의 람다값):")
        day_names = ["월", "화", "수", "목", "금", "토", "일"]
        for i, mean in enumerate(self.daily_mean_demands): print(f"  - {day_names[i]}: {mean:.2f}")
        print("="*30)

    def get_initial_state(self, start_day_index=0):
        inventory_state = np.zeros(self.tau, dtype=int)
        day_state = np.zeros(7, dtype=int)
        day_state[start_day_index] = 1
        return np.concatenate((inventory_state, day_state))

    def get_exogenous_scenario(self, length, start_day_index=0):
        demands = []
        for i in range(length):
            day_of_week = (start_day_index + i) % 7
            mean_for_today = self.daily_mean_demands[day_of_week]
            demands.append(np.random.poisson(mean_for_today)) #정수로 반환
        return np.array(demands)

    def get_next_state(self, state, action, demand):
        inventory_part = state[:self.tau]
        day_part = state[self.tau:]
        
        s_prime_inv = np.zeros_like(inventory_part)
        s_prime_inv[0] = max(0, inventory_part[0] - demand) + (inventory_part[1] if self.tau > 1 else 0)
        if self.tau > 1:
            s_prime_inv[1:-1] = inventory_part[2:]
            s_prime_inv[-1] = action

        current_day_index = np.argmax(day_part)
        next_day_index = (current_day_index + 1) % 7
        s_prime_day = np.zeros(7, dtype=int)
        s_prime_day[next_day_index] = 1
        
        return np.concatenate((s_prime_inv, s_prime_day))
    
    def get_cost(self, state, action, demand):
        inventory_part = state[:self.tau] # 비용은 재고 부분만 보고 계산
        return self.h * max(0, inventory_part[0] - demand) + self.p * max(0, demand - inventory_part[0])

    
# 2 Perishable Inventory 문제 환경 (FIFO/LIFO 비율 적용)
#State 벡터차원 및 전이 수정

# dcl_definitions_cuda.py 파일에 붙여넣을 코드
# (scipy 라이브러리는 더 이상 필요하지 않습니다)

class PerishableInventoryEnv(BaseInventoryEnv):
    def __init__(self, lifetime, lead_time, holding_cost, penalty_cost, waste_cost, 
                 mean_demand, demand_std_dev, max_action, fifo_ratio=1.0):
        self.ml, self.tau = lifetime, lead_time
        self.h, self.p, self.w = holding_cost, penalty_cost, waste_cost
        self.num_actions, self.fifo_ratio = max_action + 1, fifo_ratio
        self.state_size = self.ml + max(self.tau - 1, 0) + 7 # <<< 상태 크기 = 재고 + 파이프라인 + 요일(7)
                
        # 인스턴스 변수로 저장
        self.mean_demand = mean_demand
        self.demand_std_dev = demand_std_dev

        self.daily_mean_demands = np.maximum(1, np.random.normal(loc=mean_demand, scale=demand_std_dev, size=7))
        print("="*30 + "\nPerishable: 자동 생성된 요일별 평균 수요(포아송의 람다값):")
        day_names = ["월", "화", "수", "목", "금", "토", "일"]
        for i, mean in enumerate(self.daily_mean_demands): print(f"  - {day_names[i]}: {mean:.2f}")
        print("="*30)

    def get_initial_state(self, start_day_index=0):
        inventory_size = self.ml + max(self.tau - 1, 0)
        inventory_state = np.zeros(inventory_size, dtype=int)
        day_state = np.zeros(7, dtype=int)
        day_state[start_day_index] = 1
        return np.concatenate((inventory_state, day_state))

    def get_exogenous_scenario(self, length, start_day_index=0):
        demands = []
        for i in range(length):
            day_of_week = (start_day_index + i) % 7
            mean_for_today = self.daily_mean_demands[day_of_week]
            demands.append(np.random.poisson(mean_for_today)) #정수로 반환
        total_demands = np.array(demands)
        fifo_demands = np.random.binomial(total_demands, self.fifo_ratio)
        lifo_demands = total_demands - fifo_demands
        return list(zip(fifo_demands, lifo_demands))

    def get_next_state(self, state, action, demand_tuple):
        """
        <--- 수정 3: 논문 정의에 맞는 상태 전이 로직 구현 (노화 로직 버그 수정)
        """
        inventory_size = self.ml + max(self.tau - 1, 0)
        s_prime = np.zeros(self.state_size, dtype=int)
        
        inventory_part = state[:inventory_size]
        day_part = state[inventory_size:]
        
        on_hand_inventory = inventory_part[:self.ml]
        pipeline = inventory_part[self.ml:] if self.tau > 1 else np.array([])
        
        on_hand_after_sales, _ = self._sell_inventory(on_hand_inventory, demand_tuple)
        
        # --- 노화(Ageing) 로직 수정 ---
        # 유통기한이 2일 남았던 재고(인덱스 1)가 이제 1일 남은 재고(인덱스 0)가 됩니다.
        s_prime_inv = np.zeros_like(inventory_part)
        s_prime_inv[:self.ml-1] = on_hand_after_sales[1:self.ml]
        # --------------------------------

        arriving_stock = action if self.tau <= 1 else pipeline[0]
        s_prime_inv[self.ml - 1] += arriving_stock
        
        if self.tau > 2: s_prime_inv[self.ml:-1] = pipeline[1:]
        if self.tau > 1: s_prime_inv[-1] = action

        current_day_index = np.argmax(day_part)
        next_day_index = (current_day_index + 1) % 7
        s_prime_day = np.zeros(7, dtype=int)
        s_prime_day[next_day_index] = 1
        
        return np.concatenate((s_prime_inv, s_prime_day))

    def get_cost(self, state, action, demand_tuple):
        on_hand_inventory = state[:self.ml] # 비용은 재고 부분만 보고 계산
        inventory_after_sales, shortage = self._sell_inventory(on_hand_inventory, demand_tuple)
        perished_cost = self.w * inventory_after_sales[0]
        holding_cost = self.h * np.sum(inventory_after_sales[1:])
        penalty_cost = self.p * shortage
        return perished_cost + holding_cost + penalty_cost

    
    def _sell_inventory(self, on_hand_inventory, demand_tuple):
        demand_fifo, demand_lifo = demand_tuple
        inventory_after_sales = np.copy(on_hand_inventory)
        total_on_hand = np.sum(inventory_after_sales)
        rem_demand_fifo = demand_fifo
        for i in range(self.ml):
            can_sell = min(rem_demand_fifo, inventory_after_sales[i])
            inventory_after_sales[i] -= can_sell
            rem_demand_fifo -= can_sell
        rem_demand_lifo = demand_lifo
        for i in range(self.ml - 1, -1, -1):
            can_sell = min(rem_demand_lifo, inventory_after_sales[i])
            inventory_after_sales[i] -= can_sell
            rem_demand_lifo -= can_sell
        shortage = max(0, (demand_fifo + demand_lifo) - total_on_hand)
        return inventory_after_sales, shortage


# ===================================================================
# DCL 프레임워크 함수 및 클래스
# ===================================================================

# 알고리즘 2: SampleStartState
def sample_start_state(env, policy, L, start_day_index=0): # <<< start_day_index 추가
    state = env.get_initial_state(start_day_index)
    # <<< 시작 요일을 고려하여 시나리오 생성
    warmup_scenario = env.get_exogenous_scenario(length=L, start_day_index=start_day_index)
    for j in range(L):
        action = policy.get_action(state)
        demand = warmup_scenario[j]
        state = env.get_next_state(state, action, demand)
    return state

# Simulator의 보조 함수: 단일 롤아웃 실행
def rollout(env, start_state, initial_action, policy, H, scenario):
    total_cost = 0
    state = start_state
    
    # t=0 (첫 스텝)
    demand = scenario[0]
    total_cost += env.get_cost(state, initial_action, demand)
    state = env.get_next_state(state, initial_action, demand)
    
    # t=1 부터 H-1 까지 (get_next_state가 요일 업데이트를 자동으로 처리)
    for t in range(1, H):
        action = policy.get_action(state)
        demand = scenario[t]
        total_cost += env.get_cost(state, action, demand)
        state = env.get_next_state(state, action, demand)
    return total_cost

# 알고리즘 3: Simulator (SH + CRN)


def _run_simulation_steps(env, state, policy, H, A_r, t_r):
    """
    하나의 SH 라운드에 필요한 모든 시뮬레이션 스텝을 실행하고, 
    각 행동에 대한 비용의 '합계'를 반환하는 도우미 함수.
    (변수명을 논문 표기법에 맞춰 A_r, t_r로 변경)
    """
    round_costs = {a: 0.0 for a in A_r}
    # t_r 만큼의 외생 시나리오를 생성하여 시뮬레이션
    for _ in range(t_r):
        # CRN: 모든 행동에 동일한 시나리오를 적용하여 공정하게 비교
        scenario = env.get_exogenous_scenario(length=H)
        for action in A_r:
            round_costs[action] += rollout(env, state, action, policy, H, scenario)
    return round_costs

def simulator(env, state, policy, M, H):
    """
    논문의 변수(B_s, t_r, A_r)를 명시적으로 사용하여 가독성을 개선한 simulator 함수
    """
    # --- 1. 초기화 (Initialization) ---
    A_s = list(range(env.num_actions)) # A_s: 상태 s에서 가능한 전체 행동 집합
    if len(A_s) <= 1: return A_s[0] if A_s else 0

    
    # M은 exogenous 시나리오 수, |A_s|는 가능한 행동의 수
    B_s = M * len(A_s) 

    # num_rounds: 총 라운드 수 (K in some literature)
    # 행동 집합을 절반씩 줄여나가므로 log2(|A_s|) 만큼의 라운드가 필요
    num_rounds = math.ceil(math.log2(len(A_s))) 

    # 누적 비용과 시뮬레이션 횟수를 저장할 딕셔너리
    accumulated_costs = {a: 0.0 for a in A_s}
    
    # A_r: 현재 라운드(r)에서 살아남은 행동들의 집합
    A_r = A_s.copy() #1라운드에서는 모든 행동이 가능함으로 A_r=A_s로 정의

    # --- 2. SH 라운드 루프 (Sequential Halving Loop) ---
    for r in range(num_rounds):
        # t_r: 이번 라운드(r)에서 각 행동에 할당된 시나리오(시뮬레이션) 수
        # 안전장치: log2(|A_s|)가 0이 되는 것을 방지 (len(A_s)가 1일 때)
        log_term = math.log2(len(A_s)) if len(A_s) > 1 else 1
        denominator = len(A_r) * log_term #t_r공식에서 분모 부분을 정의.
        
        # 최소 1번은 실행하도록 max(1, ...) 처리
        t_r = max(1, math.floor(B_s / denominator)) #라운드의 횟수 함수를 만듦.

        # 복잡한 부분을 도우미 함수 호출로 단순화
        round_costs = _run_simulation_steps(env, state, policy, H, A_r, t_r)
        
        # 결과 취합
        for action in A_r:
            accumulated_costs[action] += round_costs[action]

        # --- 3. 행동 제거 (Action Elimination) ---
        if len(A_r) > 1:#라운드 결과 생존 행동이 한개 이상 남아있다면 계속 반복
            # 현재까지의 '평균' 비용을 기준으로 정렬, 근거-> pdf p.38에서 rollout평가를 평균비용으로 하고 행동 선택.
            avg_costs = {a: accumulated_costs[a] for a in A_r}
            
            # 가장 비용이 낮은 절반의 행동만 다음 라운드로 진출
            num_to_keep = math.ceil(len(A_r) / 2)
            A_r = sorted(A_r, key=lambda a: avg_costs[a])[:num_to_keep]
            
    # 모든 라운드가 끝난 후, 가장 비용이 낮은 최적의 행동을 반환
    return A_r[0]


# 단일 스레드가 실행할 작업
def collect_samples_for_thread(args):
    env, policy, num_samples, H, M, L, start_day_index = args # <<< start_day_index 추가
    local_samples = []
    current_state = sample_start_state(env, policy, L, start_day_index)
    # <<< 시작 요일을 고려하여 시나리오 생성
    current_day_idx = np.argmax(current_state[-7:])
    exogenous_scenario = env.get_exogenous_scenario(length=num_samples, start_day_index=current_day_idx)

    for k in range(num_samples):
        best_action = simulator(env, current_state, policy, M, H)
        local_samples.append((current_state.tolist(), best_action))
        demand = exogenous_scenario[k]
        current_state = env.get_next_state(current_state, best_action, demand)
        if k % 10 == 0:
            print(".", end='', flush=True)
    return local_samples

# 알고리즘 1: DCL 메인 트레이너, Deep Controlled Learning
# dcl_definitions_cuda.py 파일의 DCLTrainer 클래스 수정안

class DCLTrainer:
    """DCL 알고리즘의 전체 학습 과정을 관리합니다."""
    def __init__(self, env, initial_policy, n, N, M, H, L, w):
        self.env, self.policy = env, initial_policy
        self.n, self.N, self.M, self.H, self.L, self.w = n, N, M, H, L, w
        
    def train(self):
        trained_policies = []
        classifier_model = Classifier(self.env.state_size, self.env.num_actions)
        classifier_model.to(device)
        print(f"DCL Trainer가 {device} 장치를 사용하여 학습을 시작합니다.")

        for i in range(self.n):
            print(f"\n======= 시작: 근사 정책 반복 {i+1}/{self.n} =======")
            print(f"데이터셋 K_{i} 생성을 시작합니다 (총 {self.N}개 샘플, {self.w}개 워커)...")
            dataset_K = []
            samples_per_worker = math.ceil(self.N / self.w)
            
            # ====================================================================
            # <<< 수정된 부분 >>>
            # 리스트 컴프리헨션을 사용하여 각 워커에 대한 인자(args)를 생성합니다.
            # 이 루프가 돌 때마다 np.random.randint(0, 7)가 새로 호출되어
            # 각 워커는 서로 다른 무작위 시작 요일을 할당받게 됩니다.
            # ====================================================================
            args_list = [
                (self.env, self.policy, samples_per_worker, self.H, self.M, self.L, np.random.randint(0, 7))
                for _ in range(self.w)
            ]

            # concurrent.futures를 import해야 합니다.
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=self.w) as executor:
                futures = {executor.submit(collect_samples_for_thread, args): i for i, args in enumerate(args_list)}
                for future in as_completed(futures):
                    try:
                        dataset_K.extend(future.result())
                    except Exception as exc:
                        print(f'워커에서 예외 발생: {exc}')

            print(f"\n데이터셋 생성 완료. 총 {len(dataset_K)}개 샘플 수집.")
            # --- 정책 학습 단계 ---
            self.policy = classifier_model.train_policy(dataset_K)
            trained_policies.append(self.policy)
            
        return trained_policies
