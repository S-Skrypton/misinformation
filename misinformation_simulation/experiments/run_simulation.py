from envs.graph_environment import create_social_network
from utils.helpers import visualize_message_spread, save_paths_to_file
import random
import networkx as nx

def run_simulation(num_users):
    random.seed(42)
    G = create_social_network(num_users)
    # Simulate, visualize, and save paths as per your original main
    message_tree = simulate_message_post(G)  # You need to define or move this function appropriately
    visualize_message_spread(message_tree, G)
    save_paths_to_file(message_tree)

def simulate_message_post(G):
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

if __name__ == "__main__":
    run_simulation(500)
