import numpy as np
import gym
import os
import pickle

import gym_forestfire.agents.utils as utils
import gym_forestfire.agents.td3 as td3


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.0
    list_of_trees = []
    sum_trees = 0
    exit = 0
    
    for i in range(0, eval_episodes, 1):
        state, done = eval_env.reset(), False
        step= 0
        while not done:
            # action = policy.select_action(np.array(state))
            # action = np.random.uniform(-1, 1, 5)
            action = None
            state, reward, done, num_trees, _ = eval_env.step(action)
            avg_reward += reward
            step += 1
            if done:
                sum_trees += num_trees
                list_of_trees.append(num_trees)
                print(i, num_trees)
                if num_trees >= 0.8 * 4096:
                    exit += 1
            # if (step ==0 or step % 20 == 0) and i % 1 == 0:
            #     eval_env.render(i)
            
            

    avg_reward /= eval_episodes
    avg_trees = sum_trees/ eval_episodes
    print(list_of_trees)


    #standard deviation of list
    std = np.std(list_of_trees)
    #variance of lis
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: reward: {avg_reward:.5f}, avg trees: {avg_trees/4096:.5f} std {std/4096}, exit {exit} cops, with model {model}")
    print("---------------------------------------")
    return avg_reward

model_name = ["Crema 4-6-3 275"]
env_name = "gym_forestfire:ForestFire-v0"
seed = 1

for model in model_name:
    if os.path.exists(f"./models/{model}.pkl"):
        with open(f"./models/{model}.pkl", "rb") as f:
            policy = pickle.load(f)
            print(f"Model loaded from ./models/{model}.pkl")
# policy=None
eval_policy(policy, env_name, seed, eval_episodes=100)   