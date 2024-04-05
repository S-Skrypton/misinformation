import random
from list import LinkedList
from node import Node
from pickle import dump, load, HIGHEST_PROTOCOL  # importing pikckle for saving graph object
from sys import argv

from graph import Graph



def load_network(graph, network):
    """
    Loads a network from a given file or object(if saved earlier)
    :param graph: Graph Object
    :param network: network file
    :return: None
    """
    network = open(network, "r")  # open given file
    for line in network:  # iterate over whole file line by line
        line = line.rstrip().lstrip()  # removes any white spaces after or before the node value
        if ':' not in line:  # if line has node to add
            graph.list.add_last(LinkedList(line))  # add node to the grpah
        else:  # if line has nodes to connect
            vertices = line.split(':')  # makes list of two nodes by spliting the line on delimiter
            vertices[0] = vertices[0].rstrip().lstrip()  #
            vertices[1] = vertices[1].rstrip().lstrip()
            graph.connect(vertices[0], vertices[1])  # make a connection between node
    network.close()  # close the open file


def main():
    """
    Driver code to run the misinformation spreading simulation.
    """
    if len(argv) < 3:  # Ensure correct number of command-line arguments
        print("Usage: SocialSim -s [network_file] [prob_share]")
        return

    mode, network_file, prob_share_str = argv[1], argv[2], argv[3]

    if mode != "-s":  # Check if the simulation mode flag is correct
        print("Invalid mode. Use '-s' for simulation mode.")
        return

    try:
        prob_share = float(prob_share_str)  # Convert probability share argument to float
    except ValueError:
        print("Invalid prob_share value. Please provide a valid float.")
        return

    graph = Graph()  # Initialize the graph object

    # Load the network from the file
    load_network(graph, network_file)

    # Randomly select an initial node to start spreading misinformation
    initial_node = graph.get_random_node()
    if not initial_node:
        print("The network is empty. Cannot proceed with the simulation.")
        return

    # Output the node from which misinformation will start spreading
    print(f"Starting misinformation from node: {initial_node.get_value()} with prob_share={prob_share}")

    # Run the simulation
    spread_result = graph.spread_misinformation(initial_node, prob_share)

    # Output the simulation results
    print("\nMisinformation Spread Result:")
    for node, source in spread_result.items():
        source_str = "Initial Node" if source is None else source
        print(f"Node {node} received misinformation from {source_str}")

if __name__ == "__main__":
    main()


