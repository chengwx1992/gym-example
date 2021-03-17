#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import __init__
import torch
from rtc_env import GymEnv
from deep_rl.memory import Memory
from deep_rl.ppo_agent import PPO

def main():
    ############## Hyperparameters ##############
    env_name = "alphaRTC"
    solved_reward = 0.6         # stop training if avg_reward > solved_reward
    max_episodes = 10000        # max training episodes

    update_timestep = 4000      # update policy every n timesteps
    action_std = 0.005            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr = 3e-5                 # parameters for Adam optimizer
    betas = (0.9, 0.999)

    random_seed = None
    #############################################

    # creating environment
    #env = gym.make(env_name)
    env = GymEnv()
    state_dim = 4
    action_dim = 1

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)

    # logging variables
    running_reward = 0
    time_step = 0

    # training loop
    for i_episode in range(1, max_episodes+1):
        while time_step < update_timestep:
            done = False            
            state = torch.FloatTensor(env.reset())
            while not done:
                # Running policy_old:
                action = ppo.select_action(state, memory)
                state, reward, done, _ = env.step(action)
                state = torch.FloatTensor(state)
                # Saving reward and is_terminals:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                time_step += 1
                running_reward += reward

        # update if its time
        ppo.update(memory)
        memory.clear_memory()
        # running_reward /= time_step

        # pring Episode info
        print('Episode {} \t time step: {} \t Avg reward: {}'.format(i_episode, time_step, running_reward))

        # stop training if avg_reward > solved_reward
        if running_reward > solved_reward:
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './{}_solved_{}.pth'.format(env_name, running_reward))
        
        running_reward = 0
        time_step = 0
        
        

if __name__ == '__main__':
    main()