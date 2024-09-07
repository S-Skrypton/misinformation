import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import random


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

def plot_action_proportions(G, iteration):
    node_types = [0, 1, 2]
    action_counts_per_type = {node_type: {action: 0 for action in [0, 1, 2]} for node_type in node_types}
    type_counts = {node_type: 0 for node_type in node_types}  
    for node in G.nodes:
        node_type = G.nodes[node].get('type') 
        action = G.nodes[node].get('action') 
        if node_type in action_counts_per_type and action in action_counts_per_type[node_type]:
            action_counts_per_type[node_type][action] += 1
            type_counts[node_type] += 1 
    action_ratios_per_type = {
        node_type: {action: action_counts_per_type[node_type][action] / type_counts[node_type] if type_counts[node_type] > 0 else 0 
                    for action in [0, 1, 2]} 
        for node_type in node_types
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.25
    index = np.arange(len(node_types))
    for i, action in enumerate([0, 1, 2]):
        ratios = [action_ratios_per_type[node_type][action] for node_type in node_types]
        ax.bar(index + i * bar_width, ratios, bar_width, label=f'Action {action}')
    ax.set_xlabel('Node Types')
    ax.set_ylabel('Proportion of Actions')
    ax.set_title('Proportion of Actions by Node Types')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(node_types)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"Proportion of Actions by Node Types of iteration {iteration}.png")