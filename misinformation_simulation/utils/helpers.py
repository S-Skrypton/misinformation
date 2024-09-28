import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import networkx as nx
import numpy as np


def visualize_message_spread(message_tree, G, iteration, mode):
    plt.figure(figsize=(20, 15))
    pos = nx.multipartite_layout(message_tree, subset_key="level")
    color_map = ['gold' if G.nodes[node]['type'] == 2 else 'skyblue' if G.nodes[node]['type'] == 0 else 'pink' for node in message_tree]
    nx.draw(message_tree, pos, with_labels=True, node_size=500, node_color=color_map, font_size=8, font_weight='bold', arrowstyle='-|>', arrowsize=10)
    
    celebrity_patch = mpatches.Patch(color='gold', label='Celebrity')
    common_patch = mpatches.Patch(color='skyblue', label='Common user')
    robot_patch = mpatches.Patch(color='pink', label='Robot')
    plt.legend(handles=[celebrity_patch, common_patch, robot_patch], loc='upper right')
    
    plt.title(f"Message Spread Visualization as a Tree - Simulation {iteration} - Mode {mode}")
    plt.savefig(f"Message_Traversal_Tree_{iteration}_{mode}.png")  # Save each figure with a unique identifier
    plt.close() 

def save_paths_to_file(message_tree, iteration):
    filename = f"message_paths_{iteration}.txt" 
    root = [n for n, d in message_tree.in_degree() if d==0][0] if [n for n, d in message_tree.in_degree() if d==0] else None
    if not root:
        print("No root found in the tree.")
        return
    with open(filename, 'w') as file:
        for target in message_tree.nodes():
            if target != root:
                all_paths = list(nx.all_simple_paths(message_tree, source=root, target=target))
                for path in all_paths:
                    file.write(" -> ".join(map(str, path)) + '\n')
    print(f"All paths have been saved to {filename}.")

def save_replay_buffer_to_file(replay_buffer, filename):
    with open(filename, 'w') as f:
        for transition in replay_buffer.buffer:
            state, action, reward, next_state, done = transition
            f.write(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}\n")

def plot_action_proportions(G, iteration, message_trees):
    node_types = [0, 1, 2]
    actions = [0, 1, 2]
    action_counts_per_type = {}
    type_counts = {}

    for node_type in node_types:
        action_counts_per_type[node_type] = {}
        for action in actions:
            action_counts_per_type[node_type][action] = 0  
        type_counts[node_type] = 0  
    for message_tree in message_trees:
        for node in message_tree:
            node_type = G.nodes[node].get('type') 
            action = G.nodes[node].get('action') 
            # both are valid (node_type and action are in our lists), update the counts
            if node_type in node_types and action in actions:
                action_counts_per_type[node_type][action] += 1
                type_counts[node_type] += 1
    action_ratios_per_type = {}
    for node_type in node_types:
        action_ratios_per_type[node_type] = {}
        if type_counts[node_type] > 0:  # avoid division by zero
            for action in actions:
                action_ratios_per_type[node_type][action] = action_counts_per_type[node_type][action] / type_counts[node_type]
        else:
            for action in actions:
                action_ratios_per_type[node_type][action] = 0
    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.25
    index = np.arange(len(node_types))
    for i, action in enumerate(actions):
        ratios = []
        for node_type in node_types:
            ratios.append(action_ratios_per_type[node_type][action])
        ax.bar(index + i * bar_width, ratios, bar_width, label=f'Action {action}')
    ax.set_xlabel('Node Types')
    ax.set_ylabel('Proportion of Actions')
    ax.set_title('Average Proportion of Actions by Node Types over 100 Iterations')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(node_types)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"Proportion_of_Actions_by_Node_Types_{iteration}.png")

def plot_avg_reward(avg_reward, baseline_reward, x_range, iteration):
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, avg_reward, label='Offline Training Running Average Reward')
    plt.plot(x_range, [baseline_reward[0]]*len(x_range), linestyle='--', label='Baseline Action 0 Reward')
    plt.plot(x_range, [baseline_reward[1]]*len(x_range), linestyle='--', label='Baseline Action 1 Reward')
    plt.plot(x_range,[baseline_reward[2]]*len(x_range), linestyle='--', label='Baseline Action 2 Reward')
    plt.xlabel('Iterations')
    plt.ylabel('Average Reward')
    plt.title('Offline Training vs Baseline Policies')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Offline Training vs Baseline Policies{iteration}.png")

def plot_avg_action(G, iteration, message_trees):
    actions_by_followers = defaultdict(lambda: {'total_actions': 0, 'count': 0})
    actions_by_followings = defaultdict(lambda: {'total_actions': 0, 'count': 0})
    for message_tree in message_trees:
        for node in message_tree:
            action = G.nodes[node].get('action', 0)
            num_followers = len(G.nodes[node].get('followers', 0))
            num_followings = len(G.nodes[node].get('followings', 0))
            actions_by_followers[num_followers]['total_actions'] += action
            actions_by_followers[num_followers]['count'] += 1
            actions_by_followings[num_followings]['total_actions'] += action
            actions_by_followings[num_followings]['count'] += 1

    avg_action_by_followers = {
        k: v['total_actions'] / v['count'] for k, v in actions_by_followers.items() if v['count'] > 0
    }
    avg_action_by_followings = {
        k: v['total_actions'] / v['count'] for k, v in actions_by_followings.items() if v['count'] > 0
    }
    sorted_followers = sorted(avg_action_by_followers.keys())
    sorted_followings = sorted(avg_action_by_followings.keys())

    plt.figure(figsize=(10, 6))
    plt.plot(
        sorted_followers,
        [avg_action_by_followers[k] for k in sorted_followers],
        label='Avg Action vs Followers'
    )
    plt.xlabel('Number of Followers')
    plt.ylabel('Average Action')
    plt.title('Average Action vs Number of Followers over 100 iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Avg_Action_vs_Followers_over_100_iterations_{iteration}.png')

    plt.figure(figsize=(10, 6))
    plt.plot(
        sorted_followings,
        [avg_action_by_followings[k] for k in sorted_followings],
        label='Avg Action vs Followings',
        color='r'
    )
    plt.xlabel('Number of Followings')
    plt.ylabel('Average Action')
    plt.title('Average Action vs Number of Followings over 100 iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Avg_Action_vs_Followings_over_100_iterations_{iteration}.png')

