import rlcard
from rlcard.agents import DQNAgent, RandomAgent
from rlcard.utils import set_seed, tournament, reorganize, Logger, plot_curve
import time
# from rlcard.envs.dense_reward_gin_rummy import GinRummyDenseRewardEnv  # Import your custom environment

from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent

# import os
# import csv

# def log_performance(episode, reward, log_dir='./experiments/gin_rummy_dqn_dense/'):
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#     csv_path = os.path.join(log_dir, 'performance.csv')

#     # Check if the file exists to write the header only once
#     write_header = not os.path.exists(csv_path)

#     with open(csv_path, 'a', newline='') as file:
#         writer = csv.DictWriter(file, fieldnames=['episode', 'reward'])
#         if write_header:
#             writer.writeheader()
#         writer.writerow({'episode': episode, 'reward': reward})


def train_dqn():
    # Use the custom dense reward environment
    env = rlcard.make('gin-rummy')
    set_seed(42)

    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[256, 256, 128],
    )
    rule_agent = GinRummyNoviceRuleAgent()
    # env.set_agents([agent, RandomAgent(num_actions=env.num_actions)])
    # env.set_agents([rule_agent, RandomAgent(num_actions=env.num_actions)])
    env.set_agents([rule_agent, agent])

    logger = Logger('./experiments/gin_rummy_dqn_dense/')
    for episode in range(5000):
        trajectories, payoffs = env.run(is_training=True)
        trajectories = reorganize(trajectories, payoffs)
        for ts in trajectories[0]:
            # state, action, reward, next_state, done = ts
            # print(action, reward)
            agent.feed(ts)

        # if episode <1000:
        #     for ts in trajectories[0]:
        #         # state, action, reward, next_state, done = ts
        #         # print(action, reward)
        #         agent.feed(ts)
        # else:
        #     for ts in trajectories[1]:
        #         agent.feed(ts)
        
        # # Print rewards for each episode
        # reward0 = sum(payoffs[0]) / len(payoffs[0])
        # reward1 = sum(payoffs[1]) / len(payoffs[1])
        # print(f"Episode {episode}: Intermediate Reward Player0 = {reward0}, Player1 = {reward1}")
        
        if episode % 100 == 0:
            tournament_reward = tournament(env, 100)[1]
            print("\n")
            print(f"Episode {episode}: Tournament Reward = {tournament_reward}")
            logger.log_performance(episode, tournament_reward)



    logger.close()
    plot_curve(logger.csv_path, logger.fig_path, 'DQN')

if __name__ == '__main__':
    start_time = time.time()
    train_dqn()
    end_time = time.time()
    elapsed_time= end_time-start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")