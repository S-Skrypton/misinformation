from envs.graph_environment import create_social_network, get_state, apply_action, compute_reward, reset_adjust_rpp
from utils.helpers import visualize_message_spread, save_paths_to_file, save_replay_buffer_to_file
from agents.dqn_agent import DQN
import random
import numpy as np
import networkx as nx

def simulate_message_post(G, agent, initial_poster, mode): # !!! insert action function into this
    """Simulate message traversal in the network.
        mode = 0, 1, 2 baseline of action 0, 1, 2
        mode = r, random actions
        mode = off, offline train""" 
    message_tree = nx.DiGraph()
    queue = [(initial_poster, 0)]
    message_tree.add_node(initial_poster, level=0)
    action = 0
    # Apply action to the initial poster
    if mode == "r":
        action = random.randint(0,1)
    elif mode == "off":
        action = agent.select_action(get_state(initial_poster, G))
    elif isinstance(mode, int):
        action = mode
    else:
        print(f"Invalid mode {mode}!")
    apply_action(initial_poster, action, G)
    total_reward = compute_reward(initial_poster, None, action, G)
    G.nodes[initial_poster]['action'] = action

    while queue:
        current_node, level = queue.pop(0)
        followers = G.nodes[current_node]['followers']

        for follower in followers:
            if follower not in message_tree and random.random() < G.nodes[current_node]['adjust_rpp']:
                message_tree.add_node(follower, level=level+1)
                message_tree.add_edge(current_node, follower)
                queue.append((follower, level+1))
                if mode == "r":
                    action = random.randint(0,2)
                    state = get_state(current_node, G)
                    next_state = get_state(follower, G)
                    reward = compute_reward(current_node, follower, action, G)
                    total_reward += reward
                    done = len(G.nodes[follower]['followers']) == 0  # Adjust according to your terminal state logic
                    agent.replay_memory_buffer.add(state, action, reward, next_state, done)
                elif mode == "off":
                    follower_state = get_state(follower, G)
                    action = agent.select_action(follower_state)
                    total_reward += compute_reward(current_node, follower, action, G)
                elif isinstance(mode, int):
                    action = mode
                    total_reward += compute_reward(current_node, follower, action, G)
                else:
                    print(f"Invalid mode {mode}!")
                apply_action(follower, action, G)
                G.nodes[follower]['action'] = action
    return total_reward, message_tree


def run_simulation(num_users, iteration):   
    # Check if the graph already exists
    print(f"Creating new graph for iteration {iteration}")
    G = create_social_network(num_users)
    agent = DQN(seed=0)
    # Choose a lucky user
    initial_poster = 0
    while True:
        initial_poster = random.choice(list(G.nodes))
        # Check if the node has any followers
        if len(G.nodes[initial_poster]['followers']) > 0:
            break

    # randomly apply actions
    random_reward, message_tree = simulate_message_post(G, agent, initial_poster, "r")
    print(f"Randomly chosen actions has reward {random_reward}")
    visualize_message_spread(message_tree, G, iteration)
    save_replay_buffer_to_file(agent.replay_memory_buffer,f"replay_buffer_{iteration}.txt")
    save_paths_to_file(message_tree, iteration)
    
    # baselines
    for action in range(3):
        base_reward = []
        for trial in range(100):
            reset_adjust_rpp(G)
            single_reward,_= simulate_message_post(G, agent, initial_poster, action)
            base_reward.append(single_reward)
        avg_base = np.mean(base_reward)
        print(f"Baseline action {action} has reward {avg_base}")

    # offline training (need message tree generated by random mode)
    reset_adjust_rpp(G)
    rewards_queue = []
    # Training loop
    for i in range(2000):  # Assuming 2000 total training iterations
        agent.train(1)
        if (i + 1) % 10 == 0:
            reward, _ = simulate_message_post(G, agent, initial_poster, "off")
            rewards_queue.append(reward)
        if (i + 1) % 100 == 0:  # Evaluate every 100 iterations
            average_reward = np.mean(rewards_queue)
            print(f"Evaluation after {i + 1} iterations, the running average reward is = {average_reward}")

def run_multiple_simulations(num_users, num_simulations):
    random.seed(598)
    for i in range(1, num_simulations + 1):
        run_simulation(num_users, i)

