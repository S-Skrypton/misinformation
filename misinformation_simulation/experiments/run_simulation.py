from envs.graph_environment import create_social_network, get_state, apply_action, compute_reward, reset_graph
from utils.helpers import visualize_message_spread, save_paths_to_file, save_replay_buffer_to_file, plot_action_proportions
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
        for follower in followers:
            if follower not in message_tree and random.random() < G.nodes[current_node]['adjust_rpp']:
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
                    agent.replay_memory_buffer.add(state, action, reward, next_state, done)
                elif mode == "off":
                    follower_state = get_state(follower, G)
                    action = agent.select_action(follower_state)
                    reward = compute_reward(current_node, follower, action, G)
                    total_reward += reward
                    # Record decision
                    if record_decisions:
                        decisions.append({
                            'current_node': current_node,
                            'follower_node': follower,
                            'action': action,
                            'reward': reward,
                            'current_node_type': G.nodes[current_node]['type'],
                            'follower_node_type': G.nodes[follower]['type'],
                            'state': follower_state
                        })
                elif isinstance(mode, int):
                    action = mode
                    total_reward += compute_reward(current_node, follower, action, G)
                else:
                    print(f"Invalid mode {mode}!")
                apply_action(follower, action, G)
                G.nodes[follower]['action'] = action
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
    random_reward, message_tree = simulate_message_post(G, agent, initial_poster, "r", record_decisions=False)
    print(f"Randomly chosen actions has reward {random_reward}")
    visualize_message_spread(message_tree, G, iteration,"random")
    save_replay_buffer_to_file(agent.replay_memory_buffer,f"replay_buffer_{iteration}.txt")
    #save_paths_to_file(message_tree, iteration)

    # baselines
    for action in range(3):
        reset_graph(G)
        base_reward = []
        for trial in range(100):
            reset_graph(G)
            single_reward,message_tree= simulate_message_post(G, agent, initial_poster, action)
            base_reward.append(single_reward)
        avg_base = np.mean(base_reward)
        print(f"Baseline action {action} has average reward {avg_base}")
        visualize_message_spread(message_tree, G, iteration,f"Baseline action {action}")

    # offline training (need message tree generated by random mode)
    reset_graph(G)
    rewards_queue = []
    last_100_decisions = []
    # Training loop
    for i in range(20000):  # Assuming 2000 total training iterations
        reset_graph(G)
        agent.train(1)
        if (i + 1) % 10 == 0:
            if (i + 1) > 10:  # Start recording decisions only in the last 100 iterations
                reward, message_tree, decisions = simulate_message_post(G, agent, initial_poster, "off", record_decisions=True)
                last_100_decisions.extend(decisions)  # Collect all decisions from last 100 iterations
            else:
                reward, message_tree = simulate_message_post(G, agent, initial_poster, "off", record_decisions=False)
            rewards_queue.append(reward)
        if (i + 1) % 100 == 0:  # Evaluate every 100 iterations
            average_reward = np.mean(rewards_queue)
            print(f"Evaluation after {i + 1} iterations, the running average reward is = {average_reward}")
    visualize_message_spread(message_tree, G, iteration, "offline training")
    with open(f'last_100_decisions_iteration{iteration}.txt', 'w') as file:
        for decision in last_100_decisions:
            # Convert each decision dictionary to a string and write it to the file
            decision_str = f"{decision}\n"
            file.write(decision_str)
    state_actions = evaluate_agent(agent, G, num_states=100)
    # create_heatmap(state_actions)
    plot_action_proportions(G, iteration)
    # What needs to be added: use the agent and different random seed to run 100 plots, and gather the trend of the policy


def run_multiple_simulations(num_users, num_simulations):
    random.seed(598)
    for i in range(1, num_simulations + 1):
        run_simulation(num_users, i)


def evaluate_agent(agent, G, num_states=100):
    state_actions = []
    for _ in range(num_states):
        node = random.choice(list(G.nodes()))
        state = get_state(node, G)
        action = agent.select_action(state)
        state_actions.append((state, action))
    return state_actions

def create_heatmap(state_actions):
    # Assuming states have a specific structure, adjust indices as needed
    action_matrix = np.zeros((3, 10, 10))  # Adjust dimensions as per your state features
    counts = np.zeros((3, 10, 10))

    for state, action in state_actions:
        node_type, normalized_followers, normalized_followings = int(state[0]), int(state[1]*10), int(state[2]*10)
        action_matrix[node_type, normalized_followers, normalized_followings] += action
        counts[node_type, normalized_followers, normalized_followings] += 1

    # Average actions for heatmap
    with np.errstate(divide='ignore', invalid='ignore'):
        averaged_actions = np.nan_to_num(action_matrix / counts)

    for i, matrix in enumerate(averaged_actions):
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt=".1f")
        plt.title(f'Heatmap of Actions for Node Type {i}')
        plt.xlabel('Normalized Number of Followings')
        plt.ylabel('Normalized Number of Followers')
        plt.savefig(f"Heatmap of Actions for Node Type {i}.png")  # Save each figure with a unique identifier


