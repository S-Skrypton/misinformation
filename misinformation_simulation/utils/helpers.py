import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import random


def visualize_message_spread(message_tree, G, iteration):
    plt.figure(figsize=(20, 15))
    pos = nx.multipartite_layout(message_tree, subset_key="level")
    color_map = ['gold' if G.nodes[node]['type'] == 2 else 'skyblue' if G.nodes[node]['type'] == 0 else 'pink' for node in message_tree]
    nx.draw(message_tree, pos, with_labels=True, node_size=500, node_color=color_map, font_size=8, font_weight='bold', arrowstyle='-|>', arrowsize=10)
    
    celebrity_patch = mpatches.Patch(color='gold', label='Celebrity')
    common_patch = mpatches.Patch(color='skyblue', label='Common user')
    robot_patch = mpatches.Patch(color='pink', label='Robot')
    plt.legend(handles=[celebrity_patch, common_patch, robot_patch], loc='upper right')
    
    plt.title(f"Message Spread Visualization as a Tree - Simulation {iteration}")
    plt.savefig(f"Message_Traversal_Tree_{iteration}.png")  # Save each figure with a unique identifier
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
