from collections import deque
from envs.graph_environment import create_social_network, get_state, apply_action, compute_reward, reset_graph, cost_of_node_type
from utils.helpers import plot_avg_action, plot_avg_reward, visualize_message_spread, save_paths_to_file, save_replay_buffer_to_file, plot_action_proportions
from agents.dqn_agent import DQN
import random
import numpy as np
import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def simulate_message_post(G, agent, initial_poster, mode, record_decisions=False): # !!! insert action function into this
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

    decisions = [] #record the decisions made by the agent

    while queue:
        current_node, level = queue.pop(0)
        followers = G.nodes[current_node]['followers']
        # an extra term is needed after traversing all the followers
        temporary_buffer = []
        number_repost = 0
        for follower in followers:
            if follower not in message_tree and random.random() < G.nodes[current_node]['adjust_rpp']:
                number_repost += 1
                G.nodes[follower]['num_infected']=G.nodes[current_node]['num_infected']+1
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
                    #agent.replay_memory_buffer.add(state, action, reward, next_state, done)
                    temporary_buffer.append((state, action, reward, next_state, done))
                elif mode == "off":
                    follower_state = get_state(follower, G)
                    action = agent.select_action(follower_state)
                    reward = compute_reward(current_node, follower, action, G)
                    total_reward += reward
                    temporary_buffer.append((get_state(current_node, G), action, reward, follower_state, False))
                elif isinstance(mode, int):
                    action = mode
                    total_reward += compute_reward(current_node, follower, action, G)
                else:
                    print(f"Invalid mode {mode}!")
                apply_action(follower, action, G)
                G.nodes[follower]['action'] = action
        # compute the extra cost
        extra_cost = number_repost * cost_of_node_type(G.nodes[current_node]['type'],len(G.nodes[current_node]['followers']))
        # total_reward -= extra_cost * number_repost # needs to consider
        for experience in temporary_buffer:
            state, action, reward, next_state, done = experience
            adjusted_reward = reward - extra_cost
            if mode == "r":
                agent.replay_memory_buffer.add(state, G.nodes[current_node]['action'], adjusted_reward, next_state, done)
            elif mode == "off":
                if record_decisions:
                    decisions.append({
                        'current_node_state': state,
                        'follower_node_state': next_state,
                        'action': action,
                        'reward': adjusted_reward
                    })
    if record_decisions:
        return total_reward, message_tree, decisions
    else:
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
    random_list = []
    for trial in range(100):
        reset_graph(G)
        random_reward, message_tree = simulate_message_post(G, agent, initial_poster, "r", record_decisions=False)
        save_replay_buffer_to_file(agent.replay_memory_buffer,f"replay_buffer_{iteration}.txt")
        random_list.append(random_reward)
    print(f"Randomly chosen actions has average reward {np.mean(random_list)}")
    visualize_message_spread(message_tree, G, iteration,"random")
    #save_replay_buffer_to_file(agent.replay_memory_buffer,f"replay_buffer_{iteration}.txt")
    #save_paths_to_file(message_tree, iteration)

    # baselines
    baseline_rewards=[]
    for action in range(3):
        reset_graph(G)
        base_reward = []
        for trial in range(100):
            reset_graph(G)
            single_reward,message_tree= simulate_message_post(G, agent, initial_poster, action)
            base_reward.append(single_reward)
        avg_base = np.mean(base_reward)
        baseline_rewards.append(avg_base)
        print(f"Baseline action {action} has average reward {avg_base}")
        visualize_message_spread(message_tree, G, iteration,f"Baseline action {action}")

    # offline training (need message tree generated by random mode)
    reset_graph(G)
    reward_queue = deque(maxlen=100)
    last_100_decisions = []
    last_100_message_trees = deque(maxlen=100)
    avg_rewards = []
    # Training loop
    iteration_range = 20000
    for i in range(iteration_range):  # Assuming 2000 total training iterations
        reset_graph(G)
        agent.train(1)
        if (i + 1) % 10 == 0:
            if (i + 1) > iteration_range-100:  # Start recording decisions only in the last 100 iterations
                reward, message_tree, decisions = simulate_message_post(G, agent, initial_poster, "off", record_decisions=True)
                last_100_decisions.extend(decisions)  # Collect all decisions from last 100 iterations
                last_100_message_trees.append(message_tree)
            else:
                reward, message_tree = simulate_message_post(G, agent, initial_poster, "off", record_decisions=False)
            reward_queue.append(reward)
        # Evaluate every 100 iterations
        if (i + 1) % 100 == 0:
            average_reward = sum(reward_queue)/len(reward_queue)
            avg_rewards.append(average_reward)
            print(f"Evaluation after {i + 1} iterations, the running average reward is = {average_reward}")
    visualize_message_spread(message_tree, G, iteration, "offline training")
    x_range = range(100, iteration_range+1, 100)
    plot_avg_reward(avg_rewards, baseline_rewards, x_range, iteration)
    with open(f'last_100_decisions_iteration{iteration}.txt', 'w') as file:
        for decision in last_100_decisions:
            # Convert each decision dictionary to a string and write it to the file
            decision_str = f"{decision}\n"
            file.write(decision_str)
    plot_action_proportions(G, iteration, last_100_message_trees)
    plot_avg_action(G,iteration,last_100_message_trees)
    # What needs to be added: use the agent and different random seed to run 100 plots, and gather the trend of the policy


def run_multiple_simulations(num_users, num_simulations):
    random.seed(598)
    for i in range(1, num_simulations + 1):
        run_simulation(num_users, i)

