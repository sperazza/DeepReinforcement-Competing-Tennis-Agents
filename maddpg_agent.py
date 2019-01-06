import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ActorNN import Actor
from CriticNN import Critic
from OrnsteinNoise import OUNoise
from ReplayBuffer import ReplayBuffer


class MADDPGAgent(object):
    def __init__(self, p):
        self.p = p
        self.one_arr=np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1])
        self.zero_arr = np.array([1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.one_zero=np.array([self.one_arr,self.zero_arr])
        self.shared_memory = ReplayBuffer(self.p.ACTION_SIZE, self.p.BUFFER_SIZE, self.p.BATCH_SIZE, self.p.RANDOM_SEED,
                                          self.p.DEVICE)
        self.agents = [DDPGAgent(self.p, self.shared_memory) for _ in range(self.p.NUM_AGENTS)]

    def lr_step(self):
        for agent in self.agents:
            agent.scheduler_critic.step()
            agent.scheduler_actor.step()

    def step(self, prev_states, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in range(self.p.NUM_AGENTS):
            prev_states_both = np.hstack((self.one_zero[i], prev_states[i]))
            states_both = np.hstack((self.one_zero[i], states[i]))
            #prev_states_both = np.hstack((prev_states[0], prev_states[1]))
            #states_both = np.hstack((states[0], states[1]))
            #next_states_both = np.hstack((next_states[0], next_states[1]))

            self.shared_memory.add(prev_states_both, states_both,
                                   actions[i], rewards[i], next_states_both, dones[i])
            self.agents[i].step()


    def act(self, prev_state, state, add_noise=True):
        actions = []
        for i in range(self.p.NUM_AGENTS):
            prev_states_both=np.hstack((self.one_zero[i],prev_state[i]))
            states_both=np.hstack((self.one_zero[i],state[i]))
            actions.append(self.agents[i].act(prev_states_both, states_both, add_noise=add_noise))
        return np.array(actions)

    def reset(self):
        for agent in self.agents:
            agent.reset()


class DDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, p, shared_memory):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.p = p
        # local Replay memory
        self.memory = ReplayBuffer(self.p.ACTION_SIZE, self.p.BUFFER_SIZE, self.p.BATCH_SIZE, self.p.RANDOM_SEED,
                                   self.p.DEVICE)
        # global shared Replay memory
        self.shared_memory = shared_memory

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(p.STATE_SIZE*2, p.STATE_SIZE*2, p.ACTION_SIZE, p.RANDOM_SEED).to(p.DEVICE)
        self.actor_target = Actor(p.STATE_SIZE*2, p.STATE_SIZE*2, p.ACTION_SIZE, p.RANDOM_SEED).to(p.DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=p.LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(p.STATE_SIZE*2, p.STATE_SIZE*2, p.ACTION_SIZE, p.RANDOM_SEED).to(p.DEVICE)
        self.critic_target = Critic(p.STATE_SIZE*2, p.STATE_SIZE*2, p.ACTION_SIZE, p.RANDOM_SEED).to(p.DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=p.LR_CRITIC,
                                           weight_decay=self.p.WEIGHT_DECAY)
        self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1, gamma=0.5)
        self.scheduler_actor = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1, gamma=0.5)

        # Noise process
        self.noise = OUNoise(p.ACTION_SIZE, p.RANDOM_SEED, sigma=p.NOISE_SIGMA)

        # when you instantiate agent, make weights the same for target and local
        self.deep_copy(self.actor_target, self.actor_local)
        self.deep_copy(self.critic_target, self.critic_local)
        self.update_count = 0

    def lr_step(self):
        self.scheduler_critic.step()
        self.scheduler_actor.step()

    def step(self):  # , prev_states, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        # for prev_state, state, action, reward, next_state, done in zip(prev_states, states, actions, rewards,
        #                                                               next_states, dones):
        #        self.memory.add(prev_states, states, actions, rewards, next_states, dones)

        # Learn, if enough samples are available in memory
        self.update_count += 1
        if len(self.shared_memory) > self.p.BATCH_SIZE and self.update_count > self.p.STEPS_BEFORE_LEARN:
            self.update_count = 0
            for _ in range(self.p.NUM_LEARN_STEPS):
                experiences = self.shared_memory.sample()
                # experiences = self.memory.sample()
                self.learn(experiences, self.p.GAMMA)

    def act(self, prev_state, state, add_noise=True):
        """Returns actions for given state as per current policy."""

        prev_state = torch.from_numpy(prev_state).float().to(self.p.DEVICE)
        state = torch.from_numpy(state).float().to(self.p.DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(prev_state, state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.p.EPSILON * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        prev_states, states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models

        actions_next = self.actor_target(states, next_states)
        Q_targets_next = self.critic_target(states, next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(prev_states, states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(prev_states, states)
        actor_loss = -self.critic_local(prev_states, states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.p.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.p.TAU)

        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def deep_copy(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
