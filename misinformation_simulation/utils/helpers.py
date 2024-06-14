import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import random

def visualize_message_spread(message_tree, G):
    plt.figure(figsize=(20, 15))
    pos = nx.multipartite_layout(message_tree, subset_key="level")
    color_map = ['gold' if G.nodes[node]['type'] == 'celebrity' else 'skyblue' if G.nodes[node]['type'] == 'common' else 'pink' for node in message_tree]
    nx.draw(message_tree, pos, with_labels=True, node_size=500, node_color=color_map, font_size=8, font_weight='bold', arrowstyle='-|>', arrowsize=10)
    plt.legend(handles=[mpatches.Patch(color='gold', label='Celebrity'), mpatches.Patch(color='skyblue', label='Common user'), mpatches.Patch(color='pink', label='Robot')], loc='upper right')
    plt.title("Message Spread Visualization as a Tree")
    plt.savefig("Message_Traversal_Tree")

def save_paths_to_file(message_tree, filename="message_paths.txt"):
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
