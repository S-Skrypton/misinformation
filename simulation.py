import networkx as nx
import random
import matplotlib.pyplot as plt

def create_social_network(num_nodes):
    G = nx.DiGraph()
    # Assign types to each node and initialize properties
    for i in range(num_nodes):
        user_type = random.choice(['common', 'celebrity', 'robot'])
        if user_type == 'celebrity':
            repost_probability = 0.50
        elif user_type == 'common':
            repost_probability = 0.25
        else:
            repost_probability = 0.10
        G.add_node(i, followers=[], followings=[], type=user_type, repost_probability=repost_probability)

    # Set up followings and followers according to the new rules
    for i in G.nodes():
        user_data = G.nodes[i]
        if user_data['type'] == 'celebrity':
            # Celebrities followed by at least 40% of the users
            num_followers = max(int(0.40 * num_nodes), 1)
            followers = random.sample(list(G.nodes), num_followers)
            # Celebrities follow at most 5% of the users
            num_followings = min(int(0.05 * num_nodes), num_nodes - 1)
            followings = random.sample([n for n in G.nodes if n != i], num_followings)
        else:
            # Common users and robots followed by at most 10% of users
            num_followers = min(int(0.10 * num_nodes), num_nodes - 1)
            followers = random.sample(list(G.nodes), num_followers)
            # No specific constraint on how many these users can follow
            num_followings = random.randint(0, num_nodes - 1)
            followings = random.sample([n for n in G.nodes if n != i], num_followings)

        # Update graph with followings and followers
        for follower in followers:
            if i != follower:  # Avoid self-following
                G.add_edge(follower, i)
                G.nodes[i]['followers'].append(follower)
        
        for following in followings:
            if i != following:  # Avoid self-following
                G.add_edge(i, following)
                G.nodes[following]['followers'].append(i)

    return G

def simulate_message_post(G):
    initial_poster = random.choice(list(G.nodes()))
    message_tree = nx.DiGraph()
    queue = [(initial_poster, 0)]
    message_tree.add_node(initial_poster, level=0)
    
    while queue:
        current_node, level = queue.pop(0)
        followers = G.nodes[current_node]['followers']
        for follower in followers:
            if follower not in message_tree and random.random() < G.nodes[follower]['repost_probability']:
                message_tree.add_node(follower, level=level+1)
                message_tree.add_edge(current_node, follower)
                queue.append((follower, level+1))
    
    return message_tree

def visualize_message_spread(message_tree):
    pos = nx.multipartite_layout(message_tree, subset_key="level")
    nx.draw(message_tree, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=8, font_weight='bold', arrowstyle='-|>', arrowsize=10)
    plt.title("Message Spread Visualization as a Tree")
    plt.savefig("Message_Traversal_Tree")

if __name__ == "__main__":
    num_users = 50
    G = create_social_network(num_users)
    message_tree = simulate_message_post(G)
    visualize_message_spread(message_tree)
