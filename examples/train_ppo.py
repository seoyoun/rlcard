import os
import numpy as np
import rlcard
from rlcard.agents import DQNAgent, RandomAgent
from rlcard.agents.ppo_agent import PPOAgent
from rlcard.utils import set_seed, tournament, reorganize, Logger, plot_curve, plot_curve_winning
import time
# from rlcard.envs.dense_reward_gin_rummy import GinRummyDenseRewardEnv  # Import your custom environment

from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent
import torch

from tqdm import tqdm

# Initialize environment and agent
state_size = 260  # 5 * 52 matrix flattened
action_size = 110

total_episode = 20000
reward_stage2 = 10000
log_step = 100

buffer_size = 640
batch_size = 64
epochs = 5





env = rlcard.make('gin-rummy')

ppo_agent = PPOAgent(state_size, 
                     action_size, 
                     gamma=0.99, 
                     lam=0.95, 
                     epsilon=0.2, 
                     lr=0.0003)

## load
# checkpoint_path = './ppo/ppo_agent_episode_18000.pth'
# if os.path.exists(checkpoint_path):
#     print("exist")
#     ppo_agent.load_checkpoint(checkpoint_path)

random = RandomAgent(num_actions=env.num_actions)
rule = GinRummyNoviceRuleAgent()
agents = [
    rule,
    ppo_agent 
]
env.set_agents(agents)



# Replay buffer to store multiple trajectories
buffer = []

logger = Logger('./experiments/ppo/')

# Training loop
for episode in tqdm(range(1, reward_stage2)):
    # collect data
    trajectories = []
    state, player_id = env.reset()
    done = False
    
    while not env.is_over():
        if player_id == 0: #rule
            action = agents[0].step(state)
            state, player_id, _ = env.step(action)
        else:
            action, log_prob, value = ppo_agent.select_action(state)
            next_state, player_id, reward = env.step(action)
            if action == 1:
                break
            done = env.is_over()
            trajectories.append((state['obs'].flatten(), action, reward, log_prob, value, done))
            state = next_state

    buffer.extend(trajectories)
    if len(buffer) >= buffer_size:  # Example: Train after collecting 2048 transitions
        ppo_agent.train(buffer, batch_size=batch_size, epochs=epochs)
        buffer = []  # Clear the buffer after training
        # print("------train-------")

    if episode % log_step == 0:
        
        checkpoint_path = f'./ppo/ppo_agent_episode_{episode}.pth'
        ppo_agent.save_checkpoint(checkpoint_path)

        winning_rate = 0
        tournament_reward = [[],[]]
        for i in range(100):
            state, player_id = env.reset()
            while not env.is_over():
                if player_id == 0: #rule
                    action = agents[0].step(state)
                    state, player_id, reward = env.step(action)
                    tournament_reward[0].append(reward)
                else:
                    # print(state['obs'], type(state['obs']))
                    action, log_prob, value = ppo_agent.select_action(state)
                    next_state, player_id, reward = env.step(action)
                    if action == 1:
                        break
                    tournament_reward[1].append(reward)
                    state = next_state
            winner = env.get_winner()
            winning_rate += winner
        winning_rate /= 100
        tournament_reward = [np.mean(tournament_reward[0]),np.mean(tournament_reward[1])]
        print(f"Episode {episode}: Tournament Reward = {tournament_reward[1]} Winning = {winning_rate}")
        logger.log_performance(episode, tournament_reward[1], winning_rate)







for episode in tqdm(range(reward_stage2, total_episode)):
    # collect data
    trajectories = []
    state, player_id = env.reset()
    done = False
    
    while not env.is_over():
        if player_id == 0: #rule
            action = agents[0].step(state)
            state, player_id, _ = env.step(action)
        else:
            # print(state['obs'], type(state['obs']))
            action, log_prob, value = ppo_agent.select_action(state)
            next_state, player_id, reward = env.step_2(action)
            if action in [2,3,4,range(6,58)]:
                trajectories.append((state['obs'].flatten(), action, reward, log_prob, value, False))
                state = next_state
            else:
                state_ = state
                action_ = action
                reward_ = reward
                log_prob_ = log_prob
                value_ = value
                state = next_state
    if env.get_ender == 1:
        if env.get_winner == 0:
            trajectories.append((state_['obs'].flatten(), action_, reward_ -2, log_prob_, value_, True))
        else:
            trajectories.append((state_['obs'].flatten(), action_, reward_, log_prob_, value_, True))

    buffer.extend(trajectories)

    if len(buffer) >= buffer_size:  # Example: Train after collecting 2048 transitions
        ppo_agent.train(buffer, batch_size=batch_size, epochs=epochs)
        buffer = []  # Clear the buffer after training
        # print("------train-------")

    if episode % log_step == 0:
        
        checkpoint_path = f'./ppo/ppo_agent_episode_{episode}.pth'
        ppo_agent.save_checkpoint(checkpoint_path)

        winning_rate = 0
        tournament_reward = [[],[]]
        for i in range(100):
            state, player_id = env.reset()
            while not env.is_over():
                if player_id == 0: #rule
                    action = agents[0].step(state)
                    state, player_id, reward = env.step(action)
                    if action == 0:
                        continue
                    else:
                        tournament_reward[0].append(reward)

                else:
                    # print(state['obs'], type(state['obs']))
                    action, log_prob, value = ppo_agent.select_action(state)
                    next_state, player_id, reward = env.step_2(action)
                    if action in [2,3,4,range(6,58)]:
                        tournament_reward[1].append(reward)
                        state = next_state
                    else:
                        reward_ = reward
                        state = next_state
            winner = env.get_winner()
            # print("winner:", winner)
            winning_rate += winner

            if env.get_ender() == 1:
                if winner == 1:
                    tournament_reward[1].append(reward_)
                else:
                    tournament_reward[1].append(reward_-2)

        winning_rate /= 100
        # print("winning rate:", winning_rate)
        tournament_reward = [np.mean(tournament_reward[0]),np.mean(tournament_reward[1])]
        print(f"Episode {episode}: Tournament Reward = {tournament_reward[1]} Winning = {winning_rate}")
        logger.log_performance(episode, tournament_reward[1], winning_rate)




logger.close()
plot_curve(logger.csv_path, logger.fig_path, 'PPO')
plot_curve_winning(logger.csv_path, logger.fig_win_path, 'PPO')