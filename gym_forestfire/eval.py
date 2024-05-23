import numpy as np
import gym
import os
import pickle

import gym_forestfire.agents.utils as utils
import gym_forestfire.agents.td3 as td3


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=1):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False

        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ , _ = eval_env.step(action)
            avg_reward += reward
            eval_env.render()

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} with model {model}")
    print("---------------------------------------")
    return avg_reward

model_name = ["Prova225-PremiComplet"]
env_name = "gym_forestfire:ForestFire-v0"
seed = 0

for model in model_name:
    if os.path.exists(f"./models/{model}.pkl"):
        with open(f"./models/{model}.pkl", "rb") as f:
            policy = pickle.load(f)
            print(f"Model loaded from ./models/{model}.pkl")

    eval_policy(policy, env_name, seed, eval_episodes=20)   
