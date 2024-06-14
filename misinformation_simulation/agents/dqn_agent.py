import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import networkx as nx

# The previously defined functions
def create_social_network(num_nodes):
    """Create social network graph."""
    G = nx.DiGraph()
    num_celebrities = min(int(0.04 * num_nodes), 100)
    num_robots = min(int(0.05 * num_nodes), 500)
    num_common = num_nodes - num_celebrities - num_robots
    types = ['celebrity'] * num_celebrities + ['robot'] * num_robots + ['common'] * num_common
    random.shuffle(types)
    for i in range(num_nodes):
        user_type = types[i]
        repost_probability = 0.15 if user_type == 'celebrity' else 0.03 if user_type == 'common' else 0.01
        G.add_node(i, followers=[], followings=[], type=user_type, repost_probability=repost_probability)
    for i in G.nodes():
        user_data = G.nodes[i]
        followers = random.sample(list(G.nodes), min(int(random.uniform(0.2, 0.3) * num_nodes), 1) if user_data['type'] == 'celebrity' else min(int(random.uniform(0, 0.1) * num_nodes), 50))
        followings = random.sample([n for n in G.nodes if n != i], min(int(random.uniform(0, 0.05) * num_nodes), 10) if user_data['type'] == 'celebrity' else min(int(random.uniform(0, 0.2) * num_nodes), 50))
        for follower in followers:
            if i != follower:
                G.add_edge(follower, i)
                G.nodes[i]['followers'].append(follower)
        for following in followings:
            if i != following:
                G.add_edge(i, following)
                G.nodes[following]['followers'].append(i)
    return G

def get_state(node_id, G):
    node_data = G.nodes[node_id]
    return [
        node_data['type'],  # Type of the node
        node_data['repost_probability'],  # Current repost probability
        len(node_data['followers']) / len(G),  # Normalized number of followers
        len(node_data['followings']) / len(G)  # Normalized number of followings
    ]

def apply_action(node_id, action, G):
    """ Applies a given action to a node """
    if action == 1:
        # Block 5 posts
        G.nodes[node_id]['blocked_posts'] = 5
    elif action == 2:
        # Do nothing
        pass
    elif action == 3:
        # Label and reduce probability
        G.nodes[node_id]['repost_probability'] *= 0.3
    elif action == 4:
        # Ban all in chain - requires identifying the chain first
        pass

def cost_of_action(action):
    """Returns the cost of an action, exponentially increasing."""
    return 2 ** action

def cost_of_node_type(node_type):
    """Returns the cost associated with the node's type."""
    if node_type == 'celebrity':
        return 10
    elif node_type == 'common':
        return 5
    elif node_type == 'robot':
        return 1
    return 0

def compute_reward(node_id, action, G):
    """Computes the reward after an action."""
    action_cost = cost_of_action(action)
    node_type_cost = cost_of_node_type(G.nodes[node_id]['type'])
    reward = -(action_cost + node_type_cost)
    return reward

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
        self.replay_memory_buffer = deque(maxlen=10000)  # replay buffer
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
        if self.rng.uniform() < self.eps:  # Exploration
            action = self.rng.choice(self.output_dim)
        else:  # Exploitation
            self.dqn.eval()  # Switch to evaluation mode
            with torch.no_grad():
                state = torch.from_numpy(state).float().unsqueeze(0)
                scores = self.dqn(state)
            self.dqn.train()  # Switch back to training mode
            _, argmax = torch.max(scores.data, 1)
            action = int(argmax.numpy())
        
        return action

    def train(self, s0, a0, r, s1, done):
        """
        Train the Q network
        Args:
            s0: current state, a numpy array with size 4
            a0: current action, 0 to 3
            r: reward
            s1: next state, a numpy array with size 4
            done: done=True means that the episode terminates and done=False means that the episode does not terminate.
        """
        self.add_to_replay_memory(s0, a0, r, s1, done)
        
        if done:
            self.update_epsilon()
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
            self.eps *= 0.95