import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import networkx as nx

# QNetwork and DQN classes
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        """
        Args:
            input_dim (int): state dimension.
            output_dim (int): number of actions.
            hidden_dim (int): hidden layer dimension (fully connected layer)
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        """
        Returns a Q value
        Args:
            state (torch.Tensor): state, 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: Q values, 2-D tensor of shape (n, output_dim)
        """
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        return x


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


class DQN:
    def __init__(self, seed=None):
        self.dqn = QNetwork(4, 4, 128)  # Q network
        self.dqn_target = QNetwork(4, 4, 128)  # Target Q network
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.batch_size = 64  # Batch size
        self.output_dim = 4  # Output dimension of Q network, i.e., the number of possible actions
        self.gamma = 0.99  # Discount factor
        self.eps = 1.0  # epsilon-greedy for exploration
        self.loss_fn = torch.nn.MSELoss()  # loss function
        self.optim = torch.optim.Adam(self.dqn.parameters(), lr=0.001)  # optimizer for training
        self.replay_memory_buffer = ReplayBuffer(maxlen=10000)  # replay buffer
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)
    
    def select_action(self, state):
        """
        Returns an action for the agent to take during training process
        Args:
            state: a numpy array with size 4
        Returns:
            action: action index, 0 to 3
        """
        # if self.rng.uniform() < self.eps:  # Exploration
        #     action = self.rng.choice(self.output_dim)
        # else:  # Exploitation
        self.dqn.eval()  # Switch to evaluation mode
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0)
            scores = self.dqn(state)
        self.dqn.train()  # Switch back to training mode
        _, argmax = torch.max(scores.data, 1)
        action = int(argmax.numpy())
        
        return action

    def train(self, times):
        """
        Train the Q network
        Args:
            s0: current state, a numpy array with size 4
            a0: current action, 0 to 3
            r: reward
            s1: next state, a numpy array with size 4
            done: done=True means that the episode terminates and done=False means that the episode does not terminate.
            !!!!! either cannot repost or all the nodes are infected
        """
        # self.add_to_replay_memory(s0, a0, r, s1, done)  #!!! no need to call this function
        
        if times % 500 == 0:
            self.update_epsilon() # !!! needs modification, after several trajectories, make it slowly every 500 times
            self.target_update()
            
        if len(self.replay_memory_buffer) < self.batch_size:
            return
        
        mini_batch = self.get_random_sample_from_replay_mem()
        state_batch = torch.from_numpy(np.vstack([i[0] for i in mini_batch])).float()
        action_batch = torch.from_numpy(np.vstack([i[1] for i in mini_batch])).int()
        reward_batch = torch.from_numpy(np.vstack([i[2] for i in mini_batch])).float()
        next_state_batch = torch.from_numpy(np.vstack([i[3] for i in mini_batch])).float()
        done_list = torch.from_numpy(np.vstack([i[4] for i in mini_batch]).astype(np.uint8)).float()
        
        current_q = self.dqn(state_batch).gather(1, action_batch.view(self.batch_size, 1).type(torch.int64))
        next_q, _ = self.dqn_target(next_state_batch).max(dim=1)
        next_q = next_q.view(self.batch_size, 1)
        Q_targets = reward_batch + self.gamma * next_q * (1.0 - done_list)
        loss = self.loss_fn(current_q, Q_targets.detach())

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def add_to_replay_memory(self, state, action, reward, next_state, done):
        """
        Add samples to replay memory
        Args:
            state: current state, a numpy array with size 4
            action: current action, 0 to 3
            reward: reward
            next_state: next state, a numpy array with size 4
            done: done=True means that the episode terminates and done=False means that the episode does not terminate.
        """
        self.replay_memory_buffer.append((state, action, reward, next_state, done))

    #fill the replay buffer with graph edges
    
    def get_random_sample_from_replay_mem(self):
        """
        Random samples from replay memory without replacement
        Returns a self.batch_size length list of unique elements chosen from the replay buffer.
        Returns:
            random_sample: a list with len=self.batch_size,
                           where each element is a tuple (state, action, reward, next_state, done)
        """
        random_sample = random.sample(self.replay_memory_buffer, self.batch_size)
        return random_sample
    
    def update_epsilon(self):
        # Decay epsilon
        if self.eps >= 0.01:
            self.eps *= 0.95  # change it larger
            
    def target_update(self):
        # Update the target Q network (self.dqn_target) using the original Q network (self.dqn)
        self.dqn_target.load_state_dict(self.dqn.state_dict())

