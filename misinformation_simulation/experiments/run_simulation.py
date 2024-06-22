from envs.graph_environment import create_social_network, get_state, apply_action, compute_reward
from utils.helpers import visualize_message_spread, save_paths_to_file
from agents.dqn_agent import DQN
import random
import numpy as np
import torch
from collections import deque
import networkx as nx


def run_simulation(num_users):
    random.seed(42)
    G = create_social_network(num_users)
    # Simulate, visualize, and save paths as per your original main
    message_tree = simulate_message_post(G)  # You need to define or move this function appropriately
    visualize_message_spread(message_tree, G)
    save_paths_to_file(message_tree)

    agent = DQN(seed=0)

    # Variables for tracking rewards
    max_reward = 0
    reward_queue = deque(maxlen=100)

    # Training loop
    # for i in range(2000):
    #     node_id = random.choice(list(G.nodes))
    #     state = get_state(node_id, G)
    #     done = False
    #     episodic_reward = 0
        
    #     while not done:
    #         action = agent.select_action(np.array(state))
    #         apply_action(node_id, action, G)
    #         reward = compute_reward(node_id, action, G)
    #         next_state = get_state(node_id, G)
            
    #         # Define your own termination condition
    #         done = False
            
    #         episodic_reward += reward
    #         agent.train(np.array(state), action, reward, np.array(next_state), done)
    #         state = next_state
        
    #     reward_queue.append(episodic_reward)
        
    #     if (i + 1) % 10 == 0:
    #         print(f'Training episode {i + 1}, reward: {episodic_reward}', end='')
        
    #     if len(reward_queue) == 100:
    #         avg_reward = sum(reward_queue) / 100
    #         if (i + 1) % 10 == 0:
    #             print(f', moving average reward: {avg_reward}')
            
    #         if avg_reward > max_reward:
    #             max_reward = avg_reward
            
    #         if avg_reward >= -195:  # Adjust the threshold, first need to compute the baseline
    #             print(f"Problem solved in {i + 1} episodes")
    #             break
    #     else:
    #         if (i + 1) % 10 == 0:
    #             print('')

    print(f'Average reward over 100 episodes: {max_reward}')

def simulate_message_post(G): # !!! insert action function into this
    """Simulate message traversal in the network."""
    while True:
        initial_poster = random.choice(list(G.nodes))
        # Check if the node has any followers
        if len(G.nodes[initial_poster]['followers']) > 0:
            break
    message_tree = nx.DiGraph()
    queue = [(initial_poster, 0)]
    message_tree.add_node(initial_poster, level=0)

    while queue:
        current_node, level = queue.pop(0)
        followers = G.nodes[current_node]['followers']
        for follower in followers:
            if follower not in message_tree and random.random() < G.nodes[current_node]['repost_probability']:
                message_tree.add_node(follower, level=level+1)
                message_tree.add_edge(current_node, follower)
                queue.append((follower, level+1))

    return message_tree
