"""
Authors's Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
Paper: https://arxiv.org/abs/1802.09477
Code adapted from: https://github.com/sfujim/TD3
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight)


class Actor(nn.Module):
    """
    This class implements the Actor Network for TD3. It is a deterministic policy network that estimates the optimal action for a given state. The input to the Actor Network is the state. The output is the action. The Actor Network has two fully connected layers with ReLU activation functions. The first layer has 256 units, and the second layer has the same number of units as the action dimension. The output is scaled by the maximum action value. The Actor Network is trained using the deterministic policy gradient (DPG) algorithm. The Actor Network is optimized using the Adam optimizer.
    """

    def __init__(self, state_dim, action_dim, max_action, image_obs, cnn):
        super(Actor, self).__init__()

        self.image_obs = image_obs
        self.cnn = cnn
        self.cnn_out = state_dim * 4 * 4
        if image_obs:
            state_dim = state_dim**2

        if image_obs and cnn:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, 8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=0),
                nn.ReLU(),
            )
            self.fcn = nn.Sequential(
                nn.Linear(self.cnn_out, 512), nn.ReLU(), nn.Linear(512, 256)
            )
            self.cnn.apply(init_weights)
        else:
            self.fcn = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU())

        self.l1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        if self.image_obs and self.cnn:
            state = self.cnn(state)
            state = state.view(-1, self.cnn_out)
        state = self.fcn(state)
        a = F.relu(self.l1(state))
        return self.max_action * torch.tanh(self.l2(a))


class Critic(nn.Module):
    """
    This class implements the Critic Network for TD3. It is a Q-network that estimates the Q-value of the state-action pair. It has two Q-networks to reduce overestimation bias. The Q-networks share the same architecture. The input to the Q-network is the state-action pair. The output is the Q-value of the state-action pair. The Q-network has two fully connected layers with ReLU activation functions. The first layer has 256 units, and the second layer has 1 unit. The Q-network is trained using the mean squared error loss function. The Q-network is optimized using the Adam optimizer.
    """

    def __init__(self, state_dim, action_dim, image_obs, cnn):
        super(Critic, self).__init__()

        self.image_obs = image_obs
        self.cnn = cnn
        self.cnn_out = state_dim * 4 * 4
        if image_obs:
            state_dim = state_dim**2

        if image_obs and cnn:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, 8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=0),
                nn.ReLU(),
            )
            self.fcn_1 = nn.Sequential(
                nn.Linear(self.cnn_out + action_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
            )
            self.fcn_2 = nn.Sequential(
                nn.Linear(self.cnn_out + action_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
            )
            self.cnn.apply(init_weights)
        else:
            self.fcn_1 = nn.Sequential(
                nn.Linear(state_dim + action_dim, 256), nn.ReLU()
            )
            self.fcn_2 = nn.Sequential(
                nn.Linear(state_dim + action_dim, 256), nn.ReLU()
            )

        # Q1 architecture
        self.l1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, 1)

        # Q2 architecture
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 1)

    def forward(self, state, action):
        if self.image_obs and self.cnn:
            state = self.cnn(state)
            state = state.view(-1, self.cnn_out)
            sa = torch.cat([state, action], 1)
        else:
            sa = torch.cat([state, action], 1)

        q1 = self.fcn_1(sa)
        q1 = F.relu(self.l1(q1))
        q1 = self.l2(q1)

        q2 = self.fcn_2(sa)
        q2 = F.relu(self.l3(q2))
        q2 = self.l4(q2)
        return q1, q2

    def Q1(self, state, action):
        if self.image_obs and self.cnn:
            state = self.cnn(state)
            state = state.view(-1, self.cnn_out)
            sa = torch.cat([state, action], 1)
        else:
            sa = torch.cat([state, action], 1)

        q1 = self.fcn_1(sa)
        q1 = F.relu(self.l1(q1))
        q1 = self.l2(q1)
        return q1


class TD3(object):
    """
    This class implements the Twin Delayed Deep Deterministic Policy Gradients (TD3) algorithm. TD3 is an off-policy algorithm that learns a deterministic policy. It uses two Q-networks to reduce overestimation bias. It also uses target policy smoothing and target policy noise to improve stability. The actor network is trained using the deterministic policy gradient (DPG) algorithm. The critic network is trained using the mean squared error loss function. The actor and critic networks are optimized using the Adam optimizer. The target networks are updated using the soft update rule. The TD3 algorithm is implemented in the train method. The train method samples a batch of experiences from the replay buffer. It computes the target Q-value using the target Q-networks. It computes the current Q-value using the Q-networks. It computes the critic loss using the mean squared error loss function. It optimizes the critic network using the Adam optimizer. It computes the actor loss using the deterministic policy gradient algorithm. It optimizes the actor network using the Adam optimizer. It updates the target networks using the soft update rule. The select_action method selects an action using the actor network. The save method saves the actor and critic networks to a file. The load method loads the actor and critic networks from a file.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        image_obs=False,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        cnn=False,
    ):

        self.actor = Actor(state_dim, action_dim, max_action, image_obs, cnn).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, image_obs, cnn).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.image_obs = image_obs
        self.cnn = cnn
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        if self.image_obs and self.cnn:
            state = torch.FloatTensor(state.reshape(1, 1, 64, 64)).to(device)
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        if self.image_obs and self.cnn:
            state = state.view(-1, 1, 64, 64)
            next_state = next_state.view(-1, 1, 64, 64)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        import os

        assert os.path.exists(filename + "_critic"), "File not saved!"

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer")
        )
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
