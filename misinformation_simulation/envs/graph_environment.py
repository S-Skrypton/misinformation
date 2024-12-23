import networkx as nx
import random

def create_social_network(num_nodes):
    """Create social network graph."""
    G = nx.DiGraph()
    num_celebrities = min(int(0.05 * num_nodes), 100)
    num_robots = min(int(0.05 * num_nodes), 100)
    num_common = num_nodes - num_celebrities - num_robots
    # 2 for celebrity, 1 for robot, 0 for common
    types = [2] * num_celebrities + [1] * num_robots + [0] * num_common
    random.shuffle(types)
    for i in range(num_nodes):
        user_type = types[i]
        raw_repost_probability = 0.5 if user_type == 2 else 0.1 if user_type == 0 else 0.02
        G.add_node(i, followers=[], followings=[], type=user_type, 
                   raw_repost_probability=raw_repost_probability, 
                   adjust_rpp=raw_repost_probability, action=0,
                   num_infected=0)
    for i in G.nodes():
        user_data = G.nodes[i]
        if user_data['type'] == 2:
            followers = random.sample(list(G.nodes), max(int(random.uniform(0.1, 0.2) * num_nodes), 1))
        elif user_data['type'] == 1:
            followers = random.sample(list(G.nodes), max(int(random.uniform(0.01, 0.05) * num_nodes), 1))
        else:
            followers = random.sample(list(G.nodes), min(int(random.uniform(0, 0.005) * num_nodes), 50))
        followings = random.sample([n for n in G.nodes if n != i], min(int(random.uniform(0, 0.005) * num_nodes), 10) # Increase the number of followings of the robots
                                   if user_data['type'] == 2 else min(int(random.uniform(0, 0.02) * num_nodes), 50))
        for follower in followers:
            if i != follower:
                G.add_edge(follower, i)
                G.nodes[i]['followers'].append(follower)
                G.nodes[follower]['followings'].append(i)
        for following in followings:
            if i != following:
                G.add_edge(i, following)
                G.nodes[following]['followers'].append(i)
                G.nodes[i]['followings'].append(following)
    return G

# get state function ( add previous infected nodes)
def get_state(node_id, G):
    node_data = G.nodes[node_id]
    return [
        # add # of infected nodes before this state 
        node_data['type'],  # Type of the node
        # node_data['repost_probability'],  # Current repost probability
        len(node_data['followers']),  # Normalized number of followers
        len(node_data['followings']) ,  # Normalized number of followings
        node_data['num_infected'] # number of infected nodes
    ]

# apply action function
def apply_action(node_id, action, G):
    """ Applies a given action to a node """
    # if action == 1:
    #     # Block 5 posts
    #     G.nodes[node_id]['blocked_posts'] = 5
    if action == 0:
        # Do nothing
        pass
    elif action == 1:
        # Label and reduce probability
        G.nodes[node_id]['adjust_rpp'] *= 0.8
    elif action == 2:
        # Ban all in chain - requires identifying the chain first (don't let any reposts from this node go through)
        # skip this node and all its followers
        G.nodes[node_id]['adjust_rpp'] *= 0.5
        # stack = [node_id]
        # while stack:
        #     current_node = stack.pop()
        #     G.nodes[current_node]['repost_probability'] = 0
        #     followers = G.nodes[current_node]['followers']
        #     stack.extend(followers)
    else:
        print(f"print invalid action! {action}")

# Cost functions
def cost_of_action(action, num_followers):
    """Returns the cost of an action, exponentially increasing."""
    if action == 0:
        return 0
    elif action == 1:
        return 1
    elif action == 2:
        return 10

def cost_of_node_type(node_type,num_followers):
    """Returns the cost associated with the node's type."""
    if node_type == 2:
        return 0.1*num_followers
    elif node_type == 0:
        return 0.1*num_followers
    elif node_type == 1:
        return 0.01*num_followers
    return 0

def compute_reward(current_node, next_node, action, G):
    """Computes the reward after an action based on current and next node."""
    action_cost = cost_of_action(action,len(G.nodes[current_node]['followers']))
    current_node_type_cost = cost_of_node_type(G.nodes[current_node]['type'],len(G.nodes[current_node]['followers']))
    # For root node
    next_node_type_cost = 0
    if next_node is not None:    
        next_node_type_cost = cost_of_node_type(G.nodes[next_node]['type'],len(G.nodes[next_node]['followers']))

    # Assuming you want to incorporate the type cost of both nodes
    reward = -(action_cost + current_node_type_cost + next_node_type_cost)
    return reward

def reset_graph(G):
    """
    Reset adjust_rpp attribute for each node in the graph to match raw_repost_probability.
    """
    for i in G.nodes():
        G.nodes[i]['adjust_rpp'] = G.nodes[i]['raw_repost_probability']
        G.nodes[i]['num_infected']=0

# DONE 