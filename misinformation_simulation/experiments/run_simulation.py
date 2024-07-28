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

def simulate_message_post(G, agent, initial_poster, mode): # !!! insert action function into this
    """Simulate message traversal in the network.
        mode = 0, random action
        mode = 1, baseline test
        mode = 2, offline train"""
    while True:
        initial_poster = random.choice(list(G.nodes))
        # Check if the node has any followers
        if len(G.nodes[initial_poster]['followers']) > 0:
            break
    message_tree = nx.DiGraph()
    queue = [(initial_poster, 0)]
    message_tree.add_node(initial_poster, level=0)
    # Apply a random action to the initial poster
    action = random.randint(0,1)  # Assume actions are numbered 1 to 3
    apply_action(initial_poster, action, G)
    initial_reward = compute_reward(initial_poster, None, action, G)
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
                action = random.randint(0, 2)
                apply_action(follower, action, G)
                G.nodes[follower]['action']=action
                
                state = get_state(current_node, G)
                next_state = get_state(follower, G)
                reward = compute_reward(current_node, follower, action, G)
                initial_reward += reward
                done = len(G.nodes[follower]['followers']) == 0  # Adjust according to your terminal state logic
                agent.replay_buffer.add(state, action, reward, next_state, done)
    print(f"initial_reward is {initial_reward}")
    return initial_reward, message_tree



def run_simulation(num_users, iteration):   
    # Check if the graph already exists
    print(f"Creating new graph for iteration {iteration}")
    G = create_social_network(num_users)
    agent = DQN(seed=0)
    # message_tree = simulate_message_post(G)
    initial_reward, message_tree = simulate_message_post(G, agent, initial_poster, 0)
    visualize_message_spread(message_tree, G, iteration)
    save_replay_buffer_to_file(agent.replay_memory_buffer,f"replay_buffer_{iteration}.txt")
    save_paths_to_file(message_tree, iteration)
    initial_poster=[n for n, d in message_tree.in_degree() if d==0][0] if [n for n, d in message_tree.in_degree() if d==0] else None
    rewards_queue = []
    # rewards_queue.append(initial_reward)
    # Training loop
    for i in range(2000):  # Assuming 2000 total training iterations
        agent.train(1)
        if (i + 1) % 10 == 0:
            reward = simulate_spread(G, agent, initial_poster)
            rewards_queue.append(reward)
        if (i + 1) % 100 == 0:  # Evaluate every 100 iterations
            average_reward = np.mean(rewards_queue)
            print(f"Evaluation after {i + 1} iterations: running average is = {average_reward}")

def run_multiple_simulations(num_users, num_simulations):
    random.seed(598)
    for i in range(1, num_simulations + 1):
        run_simulation(num_users, i)

