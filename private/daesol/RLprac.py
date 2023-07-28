import numpy as np

# 도시와 주유소 정보 (예시로 5개의 도시와 1개의 주유소를 생성)
num_cities = 7
num_fuelpumps = 1
max_fuel = 100

# 고정된 주유소 좌표
fuelpump_coordinates = [(50, 50)]

# Q-learning 파라미터
num_episodes = 1000
learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.2

# Q 테이블 초기화
Q_table = np.zeros((num_cities, num_fuelpumps+1, num_cities))

# 도시 좌표 정보 (수정된 좌표로 변경)
city_coordinates = [[50, 50],
                    [83, 93],
                    [6.5, 30],
                    [94, 38],
                    [23, 57.5],
                    [43, 59.5],
                    [82, 3]]

def calculate_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(num_cities)
    else:
        return np.argmax(Q_table[state[0], state[1], :])

def update_q_table(state, action, next_state, reward, fuel, done):
    if done:
        Q_table[state[0], state[1], action] = reward
    else:
        max_next_action = np.argmax(Q_table[next_state[0], next_state[1], :])
        Q_table[state[0], state[1], action] = (1 - learning_rate) * Q_table[state[0], state[1], action] + \
                                              learning_rate * (reward + discount_factor * Q_table[next_state[0], next_state[1], max_next_action])

def run_episode():
    state = (np.random.randint(0, num_cities), 0)  # 튜플로 상태를 표현: (도시 인덱스, 주유소 방문 횟수)
    fuel = max_fuel
    for _ in range(num_cities):
        action = choose_action(state, epsilon)
        if fuel <= 0:
            fuel = max_fuel
            if state[0] in range(num_fuelpumps):
                fuel_reward = 10
            else:
                fuel_reward = -10
        else:
            fuel_reward = 0
        
        fuel -= calculate_distance(city_coordinates[state[0]], city_coordinates[action])
        next_state = (action, state[1])
        done = (next_state[0] == 0)
        
        # 현재 상태, 액션으로 Q 테이블 업데이트
        update_q_table(state, action, next_state, fuel_reward, fuel, done)
        
        state = next_state
        if done:
            break

# Q-learning 알고리즘 학습
for episode in range(num_episodes):
    run_episode()

# 학습된 Q 테이블로 최적 경로 추론
def find_optimal_path(start_city):
    current_city = (start_city, 0)
    fuel = max_fuel
    path = [current_city[0]]
    while current_city[0] != 0:
        action = np.argmax(Q_table[current_city[0], current_city[1], :])
        next_city = (action, current_city[1])
        if fuel <= 0:
            fuel = max_fuel
            if current_city[0] in range(num_fuelpumps):
                fuel_reward = 10
            else:
                fuel_reward = -10
        else:
            fuel_reward = 0
        
        fuel -= calculate_distance(city_coordinates[current_city[0]], city_coordinates[next_city[0]])
        path.append(next_city[0])
        current_city = next_city
    return path

start_city = 0
optimal_path = find_optimal_path(start_city)
print("Optimal path:", optimal_path)