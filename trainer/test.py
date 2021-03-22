import __init__
import torch
import matplotlib.pyplot as plt
import numpy as np
from rtc_env import GymEnv
from deep_rl.storage import Storage
from deep_rl.actor_critic import ActorCritic

def draw(record_action, record_state, path):
    length = len(record_action)
    plt.subplot(411)
    plt.plot(range(length), record_action)
    plt.xlabel('episode')
    plt.ylabel('action')
    ylabel = ['receiving rate', 'delay', 'packet loss']
    record_state = [t.numpy() for t in record_state]
    record_state = np.array(record_state)
    for i in range(3):
        plt.subplot(411+i+1)
        plt.plot(range(length), record_state[:,i])
        plt.xlabel('episode')
        plt.ylabel(ylabel[i])
    plt.tight_layout()
    plt.savefig("{}test_result.jpg".format(path))

def test(model, data_path):
    env = GymEnv()
    record_reward = []
    record_state = []
    record_action = []
    episode_reward  = 0
    time_step = 0
    tmp = model.random_action
    model.random_action = False
    while time_step < max_num_steps:
        done = False            
        state = torch.Tensor(env.reset())
        while not done:
            action, _, _ = model.forward(state)
            state, reward, done, _ = env.step(action)
            state = torch.Tensor(state)
            record_state.append(state)
            record_reward.append(reward)
            record_action.append(action)
            time_step += 1
    model.random_action = True
    draw(record_action, record_state, data_path)

# Example
# Model path and data path
model_path = f'./data/model/pretrained_model.pth'
data_path = f'./data/test/'

env_name = "alphaRTC"
max_num_steps = 1000
state_dim = 4
action_dim = 1
model = ActorCritic(state_dim, action_dim)
model.load_state_dict(torch.load(model_path))
test(model, data_path)