import numpy as np
import rlcard
from rlcard.agents import DQNAgent, RandomAgent
from rlcard.agents.ppo_agent import PPOAgent
from rlcard.utils import set_seed, tournament, reorganize, Logger, plot_curve
import time
# from rlcard.envs.dense_reward_gin_rummy import GinRummyDenseRewardEnv  # Import your custom environment

from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent
import torch

from tqdm import tqdm

# Initialize environment and agent
state_size = 260  # 5 * 52 matrix flattened
action_size = 110


env = rlcard.make('gin-rummy')

ppo_agent = PPOAgent(state_size, action_size)
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
for episode in tqdm(range(1, 10000)):
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
            next_state, player_id, reward = env.step(action)
            done = env.is_over()
            trajectories.append((state['obs'].flatten(), action, reward, log_prob, value, done))
            state = next_state

    # Train the agent after collecting trajectories
    #states, actions, rewards, log_probs, values, dones = zip(*trajectories)
       # Train after collecting enough transitions
    if len(buffer) >= 2048:  # Example: Train after collecting 2048 transitions
        ppo_agent.train(buffer, batch_size=64)
        buffer = []  # Clear the buffer after training

    if episode % 100 == 0:
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
                    tournament_reward[1].append(reward)
                    state = next_state
        tournament_reward = [np.mean(tournament_reward[0]),np.mean(tournament_reward[1])]
        print(f"Episode {episode}: Tournament Reward = {tournament_reward[1]}")



        logger.log_performance(episode, tournament_reward[1])
        # print("0----------------------0")



logger.close()
plot_curve(logger.csv_path, logger.fig_path, 'PPO')