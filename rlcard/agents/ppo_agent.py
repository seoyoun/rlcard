import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



def compute_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    next_value = 0  # Terminal states have zero value

    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_value = 0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        next_value = values[t]

    return torch.FloatTensor(advantages)


class PPOAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lam=0.95, epsilon=0.2, lr=0.0003):
        """
        Initialize the PPO agent.

        Args:
            state_size (int): Size of the flattened state (5*52 = 260 for your Gin Rummy task).
            action_size (int): Total number of actions.
            gamma (float): Discount factor.
            lam (float): GAE lambda.
            epsilon (float): Clipping parameter for PPO.
            lr (float): Learning rate.
        """
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon

        # Initialize policy network
        self.policy_net = PPOPolicy(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def mask_policy(self, policy, legal_actions):
        mask = torch.zeros_like(policy)
        mask[legal_actions] = 1.0
        masked_policy = policy * mask
        return masked_policy / masked_policy.sum(dim=-1, keepdim=True)

    def select_action(self, state):
        """
        Select an action based on the policy and legal actions.

        Args:
            state (dict): The current state from the environment.

        Returns:
            action (int): The selected action ID.
            log_prob (float): Log-probability of the selected action.
            value (float): Estimated value of the state.
        """
        state_tensor = torch.FloatTensor(state['obs'].flatten()).unsqueeze(0)
        policy, value = self.policy_net(state_tensor)

        # Mask the policy for legal actions
        legal_actions = list(state['legal_actions'].keys())
        masked_policy = self.mask_policy(policy.squeeze(), legal_actions)

        # Sample an action
        action = torch.multinomial(masked_policy, 1).item()
        log_prob = torch.log(masked_policy[action])

        return action, log_prob, value.item()

    def train(self, trajectories, epochs=10):
        """
        Train the agent using collected trajectories.

        Args:
            trajectories (list): List of collected trajectories. Each trajectory contains:
                - states: List of states.
                - actions: List of actions.
                - rewards: List of rewards.
                - log_probs: List of log-probabilities of actions.
                - values: List of state values.
                - dones: List of done flags.
            epochs (int): Number of epochs for PPO updates.
        """
        states, actions, rewards, log_probs, values, dones = zip(*trajectories)

        # Flatten trajectories
        states = torch.FloatTensor(np.vstack(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        log_probs = torch.FloatTensor(log_probs)
        values = torch.FloatTensor(values)
        dones = np.array(dones, dtype=bool)

        # Compute advantages and returns
        advantages = compute_advantages(rewards, values, dones, self.gamma, self.lam)
        returns = advantages + values

        for _ in range(epochs):
            # Get current policy and value predictions
            policy, value_preds = self.policy_net(states)

            # Compute new log-probabilities for the selected actions
            new_log_probs = torch.log(policy.gather(1, actions).squeeze())

            # Compute probability ratio
            ratios = torch.exp(new_log_probs - log_probs)

            # Compute the clipped objective
            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.mean(torch.min(surrogate1, surrogate2))

            # Compute value loss (MSE)
            value_loss = torch.mean((returns - value_preds.squeeze()) ** 2)

            # Entropy bonus for exploration
            entropy = -torch.sum(policy * torch.log(policy + 1e-10), dim=1).mean()
            entropy_bonus = 0.01 * entropy

            # Total loss
            total_loss = policy_loss + 0.5 * value_loss - entropy_bonus

            # Update policy network
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

class PPOPolicy(nn.Module):
    def __init__(self, state_size, action_size):
        super(PPOPolicy, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.policy_head = nn.Linear(64, action_size)
        self.value_head = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        policy = self.softmax(self.policy_head(x))  # Probabilities
        value = self.value_head(x)  # State value
        return policy, value