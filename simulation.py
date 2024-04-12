import networkx as nx
import random
import matplotlib.pyplot as plt

def create_social_network(num_nodes):
    """Create social network graph."""
    G = nx.DiGraph()
    # at most min(10%, 200) celebrities
    num_celebrities = min(int(0.01 * num_nodes), 100) 
    # at most min(20%, 400) robots
    num_robots = min(int(0.05 * num_nodes), 500)       
    num_common = num_nodes - num_celebrities - num_robots
    types = ['celebrity'] * num_celebrities + ['robot'] * num_robots + ['common'] * num_common
    random.shuffle(types) 
    # assign user type and repost probability
    for i in range(num_nodes):
        user_type = types[i]
        if user_type == 'celebrity':
            repost_probability = 0.15
        elif user_type == 'common':
            repost_probability = 0.05
        else:
            repost_probability = 0.01
        G.add_node(i, followers=[], followings=[], type=user_type, repost_probability=repost_probability)

    # set connections
    for i in G.nodes():
        user_data = G.nodes[i]
        if user_data['type'] == 'celebrity':
            # celebrities followed by at least 20% and at most 30% of the users
            num_followers = min(int(random.uniform(0.2, 0.3) * num_nodes), 1)
            followers = random.sample(list(G.nodes), num_followers)
            # celebrities follow at most min(5%, 10) of the users
            num_followings = min(int(random.uniform(0, 0.05) * num_nodes), 10)
            followings = random.sample([n for n in G.nodes if n != i], num_followings)
        else:
            # common users and robots followed by at most min(10%, 100)of users 
            num_followers = min(int(random.uniform(0, 0.1) * num_nodes), 50)
            followers = random.sample(list(G.nodes), num_followers)
            # follow at most min(20%, 50) of the users
            num_followings = min(int(random.uniform(0, 0.2) * num_nodes), 50)
            followings = random.sample([n for n in G.nodes if n != i], num_followings)

        # update the directed graph with followings and followers
        for follower in followers:
            if i != follower:  # avoid self-following
                G.add_edge(follower, i)
                G.nodes[i]['followers'].append(follower)
        
        for following in followings:
            if i != following: 
                G.add_edge(i, following)
                G.nodes[following]['followers'].append(i)

    return G

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

def visualize_message_spread(message_tree):
    pos = nx.multipartite_layout(message_tree, subset_key="level")
    nx.draw(message_tree, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=8, font_weight='bold', arrowstyle='-|>', arrowsize=10)
    plt.title("Message Spread Visualization as a Tree")
    plt.savefig("Message_Traversal_Tree")

def save_paths_to_file(message_tree, filename="message_paths.txt"):
    """Save all paths from the root to leaf nodes in the message_tree to a file."""
    root = [n for n, d in message_tree.in_degree() if d==0]  # find the root node
    if not root:
        print("No root found in the tree.")
        return
    root = root[0]

    with open(filename, 'w') as file:
        for target in message_tree.nodes():
            if target == root:
                continue  # Skip the root itself
            all_paths = list(nx.all_simple_paths(message_tree, source=root, target=target))
            for path in all_paths:
                path_str = " -> ".join(map(str, path))
                file.write(path_str + '\n')

    print(f"All paths have been saved to {filename}.")

if __name__ == "__main__":
    random.seed(42)
    num_users = 500
    G = create_social_network(num_users)
    message_tree = simulate_message_post(G)
    visualize_message_spread(message_tree)
    save_paths_to_file(message_tree)
