import networkx as nx
import random

def create_social_network(num_nodes):
    """Create social network graph."""
    G = nx.DiGraph()
    num_celebrities = min(int(0.04 * num_nodes), 100)
    num_robots = min(int(0.05 * num_nodes), 500)
    num_common = num_nodes - num_celebrities - num_robots
    types = ['celebrity'] * num_celebrities + ['robot'] * num_robots + ['common'] * num_common
    random.shuffle(types)
    for i in range(num_nodes):
        user_type = types[i]
        repost_probability = 0.15 if user_type == 'celebrity' else 0.03 if user_type == 'common' else 0.01
        G.add_node(i, followers=[], followings=[], type=user_type, repost_probability=repost_probability)
    for i in G.nodes():
        user_data = G.nodes[i]
        followers = random.sample(list(G.nodes), min(int(random.uniform(0.2, 0.3) * num_nodes), 1) if user_data['type'] == 'celebrity' else min(int(random.uniform(0, 0.1) * num_nodes), 50))
        followings = random.sample([n for n in G.nodes if n != i], min(int(random.uniform(0, 0.05) * num_nodes), 10) if user_data['type'] == 'celebrity' else min(int(random.uniform(0, 0.2) * num_nodes), 50))
        for follower in followers:
            if i != follower:
                G.add_edge(follower, i)
                G.nodes[i]['followers'].append(follower)
        for following in followings:
            if i != following:
                G.add_edge(i, following)
                G.nodes[following]['followers'].append(i)
    return G

# get state function
def get_state(node_id, G):
    node_data = G.nodes[node_id]
    return [
        node_data['type'],  # Type of the node
        node_data['repost_probability'],  # Current repost probability
        len(node_data['followers']) / len(G),  # Normalized number of followers
        len(node_data['followings']) / len(G)  # Normalized number of followings
    ]

# apply action function
def apply_action(node_id, action, G):
    """ Applies a given action to a node """
    if action == 1:
        # Block 5 posts
        G.nodes[node_id]['blocked_posts'] = 5
    elif action == 2:
        # Do nothing
        pass
    elif action == 3:
        # Label and reduce probability
        G.nodes[node_id]['repost_probability'] *= 0.3
    elif action == 4:
        # Ban all in chain - requires identifying the chain first
        pass

def compute_reward(node_id, action, G):
    """ Computes the reward after an action """
    # Define and compute the reward based on action specifics
    return reward