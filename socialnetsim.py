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
    Contains driver code for the project
    :return: None
    """
    graph = Graph()  # createsnew graph object
    pro_like = 0  # intilize probabily of liking a post to zeero
    pro_follow = 0  # intilize probabily of folloing a person a post to zero
    time = 1  # will tell time sep number
    time_step = []  # this will contain list of nodes that will be affected with each timestep
    try:
        if not len(argv) >= 2:  # if we dont have required arguments than show usage information
            print("Usage: SocialSim -s [network_file] [event_file] [prob_like] [prob_follow]")
            print("-s : simulation mode")
            return
        load_network(graph, argv[2])
        pro_follow = int(argv[5])
        pro_like = int(argv[4])
        misinfo = open(argv[3], "r")
        node = graph.list.find_node(line[1])
        if node is not None:
            node.posts.append(line[2])
            time_step.append(node)
            if time_step:
                if pro_like >= 0.5:
                    n = time_step[0].value.head
                    while n:
                        check += 1
                        node.likes += 1
                        if pro_follow >= 0.5:
                            n1 = graph.list.find_node(str(n.value))
                            if n1 is not None:
                                n1 = n1.value.head
                                while n1:
                                    if check > 100:
                                        break
                                    check += 1
                                    if node.value.find_node(str(n1.value)) is None:
                                        graph.connect(str(node), str(n1.value))

                                    n1 = n1.next_node
                        n = n.next_node
                    time_step = time_step[1:]
            time += 1
        file = open("simulation_Log.txt", "w")
        likes = []
        person = graph.list.head
        file.write("\t\t\tPeople in order of popularity.\n")
        while person:
            likes.append((person.likes, person.value, person.posts, person.fol, person.flrs))
            person = person.next_node
        likes.sort(key=lambda t: t[0], reverse=True)
        for like in likes:
            file.write("\t\tPeople: " + str(like[1].value) + " Total Likes: " + str(like[0]) + "\n")
        file.write("\t\t\tPosts in order of popularity.\n")
        for like in likes:
            post = like[2]
            for p in post:
                file.write("\t\tPost: " + str(p) + "\n")
        for like in likes:
            file.write("\t\t" + str(like[1].value) + "'s record:\n\t\t\tPosts # " + str(len(like[2])) + "\n")
            file.write("\t\t\tFollowers # " + str(like[4]) + "\n\t\t\tFollowing # " + str(like[3]) + "\n")
        file.close()
    except Exception as ex:
        print(ex)


if __name__ == "__main__": main()