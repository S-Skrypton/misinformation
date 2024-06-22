from envs.graph_environment import create_social_network, get_state, apply_action, compute_reward
from utils.helpers import visualize_message_spread, save_paths_to_file, save_replay_buffer_to_file
from agents.dqn_agent import DQN
import random
import numpy as np
import torch
from collections import deque
import networkx as nx
import json
import os


def run_simulation(num_users, iteration):
    
    filename = f"network_simulation_{iteration}.json"
    
    # Check if the graph already exists
    if os.path.exists(filename):
        print(f"Loading existing graph for iteration {iteration}")
        G = load_graph_from_json(filename)
    else:
        print(f"Creating new graph for iteration {iteration}")
        G = create_social_network(num_users)
        save_graph_to_json(G, filename)  # Save the newly created graph to a JSON file
    agent = DQN(seed=0)
    # message_tree = simulate_message_post(G)
    message_tree = simulate_message_post(G, agent.replay_memory_buffer)
    visualize_message_spread(message_tree, G, iteration)
    save_replay_buffer_to_file(agent.replay_memory_buffer,"replay_buffer.txt")
    save_paths_to_file(message_tree, iteration)
    # # Variables for tracking rewards
    # max_reward = 0
    # reward_queue = deque(maxlen=100)


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

    # print(f'Average reward over 100 episodes: {max_reward}')

def run_multiple_simulations(num_users, num_simulations):
    random.seed(598)
    for i in range(1, num_simulations + 1):
        run_simulation(num_users, i)

def simulate_message_post(G, replay_buffer): # !!! insert action function into this
    """Simulate message traversal in the network."""
    while True:
        initial_poster = random.choice(list(G.nodes))
        # Check if the node has any followers
        if len(G.nodes[initial_poster]['followers']) > 0:
            break
    message_tree = nx.DiGraph()
    queue = [(initial_poster, 0)]
    message_tree.add_node(initial_poster, level=0)

    # Apply a random action to the initial poster
    action = random.randint(1, 2)  # Assume actions are numbered 1 to 3
    apply_action(initial_poster, action, G)
    G.nodes[initial_poster]['action'] = action

    while queue:
        current_node, level = queue.pop(0)
        followers = G.nodes[current_node]['followers']

        for follower in followers:
            if follower not in message_tree and random.random() < G.nodes[current_node]['repost_probability']:
                message_tree.add_node(follower, level=level+1)
                message_tree.add_edge(current_node, follower)
                queue.append((follower, level+1))
                # Apply a random action to the follower
                action = random.randint(1, 4)
                apply_action(follower, action, G)
                G.nodes[follower]['action']=action
                
                state = get_state(current_node, G)
                next_state = get_state(follower, G)
                reward = compute_reward(current_node, follower, action, G)
                done = len(G.nodes[follower]['followers']) == 0  # Adjust according to your terminal state logic
                replay_buffer.add(state, action, reward, next_state, done)
    return message_tree

def save_graph_to_json(graph, filename):
    data = nx.node_link_data(graph)  # convert graph to node-link format which is suitable for JSON
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_graph_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return nx.node_link_graph(data)  # create a NetworkX graph from node-link data

