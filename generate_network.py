import random
import string

def generate_unique_node_names(num_nodes):
    """
    Generate unique node names using letters. This function ensures uniqueness
    but is limited by the number of letters available. For more nodes, consider
    a different naming strategy.

    :param num_nodes: The number of unique node names to generate.
    :return: A list of unique node names.
    """
    return ['Node_' + ''.join(random.choices(string.ascii_letters, k=5)) for _ in range(num_nodes)]

def generate_network(num_nodes, filename='network.txt'):
    """
    Generates a network file with a specified number of nodes and randomly established connections.

    :param num_nodes: The number of nodes to generate in the network.
    :param filename: The name of the file to which the network will be saved.
    """
    nodes = generate_unique_node_names(num_nodes)
    connections = set()

    # Ensure at least one connection per node for a more interconnected graph
    for node in nodes:
        # Determine a random number of connections for this node (at least 1)
        num_connections = random.randint(1, min(len(nodes) - 1, 20))  # Assuming a max of 5 connections for simplicity
        connections.update((node, random.choice([n for n in nodes if n != node])) for _ in range(num_connections))

    # Write the network to a file
    with open(filename, "w") as file:
        # Write nodes (not strictly necessary for loading, but good for verification)
        for node in nodes:
            file.write(node + "\n")
        
        # Write connections
        for source, target in connections:
            file.write(f"{source}:{target}\n")

if __name__ == "__main__":
    # Example usage
    generate_network(30, "random_network.txt")
