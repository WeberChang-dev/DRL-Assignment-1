import numpy as np
import pickle
from simple_custom_taxi_env import SimpleTaxiEnv

with open('policy_table.pkl', 'rb') as f:
    policy_table = pickle.load(f)

def softmax(x):
        exp_x = np.exp(x - np.max(x)) 
        return exp_x / exp_x.sum()

def get_state(obs):
    _,_,_,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs
    return (obstacle_south, obstacle_north, obstacle_east, obstacle_west, passenger_look, destination_look)

def get_action(obs):
    state = get_state(obs)
    return np.random.choice([0, 1, 2, 3, 4, 5], p=softmax(np.array(policy_table[state])))

def run_train(episodes=5000, gamma=0.99, alpha=0.1):
    env_config = {
        "fuel_limit": 5000
    }

    env = SimpleTaxiEnv(**env_config)
    sum = 0
    
    for i in range(episodes):

        obs, _ = env.reset()
        total_reward = 0
        done = False
        first_pickup = True
        step_count = 0
        trajectory = []

        while not done:
            
            state = get_state(obs)
            action = get_action(obs)
            obs, reward, done, _ = env.step(action)
            if first_pickup and env.passenger_picked_up:
                first_pickup = False
                reward += 25
            total_reward += reward
            step_count += 1
            trajectory.append((state, action, reward))

        if step_count < 5000:
            print(f"Episode {i} Success! Reward: {total_reward}")
            G = 0
            for t in reversed(range(len(trajectory))):
                state, action, reward = trajectory[t]
                reward *= 0.0001
                G = reward + gamma * G  

                grad = np.zeros(6)
                grad[action] = 1
                grad -= softmax(policy_table[state])
                policy_table[state] += alpha * G * grad
        else:
            print(f"Episode {i} Failed! Reward: {total_reward}")
        sum += total_reward
        
    for key, value in policy_table.items():
        new_val = [float(x) for x in value]
        policy_table[key] = new_val

    with open('tmp.pkl', 'wb') as f:
        pickle.dump(policy_table, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Average reward: {sum / episodes}")

run_train()