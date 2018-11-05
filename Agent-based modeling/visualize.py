import matplotlib.pyplot as plt
import networkx as nx

def visualize(network, class_id, step, simulated=True):
    """
    Args:
        network: NetworkX DiGraph
        class_id: the unique id of class
        step: a number between 0 and NUM_STEP
        simulated: False if it is a real network
    """
    
    plt.figure(figsize=(25, 8))
    plt.tight_layout()
    plt.suptitle('Simulated on Class {}, step {}'.format(class_id, step), fontsize = 30)
    
    females = [idx for idx, sex in nx.get_node_attributes(network, 'sex').items() if sex == 'female']
    males = [idx for idx, sex in nx.get_node_attributes(network, 'sex').items() if sex == 'male']
    male_network = network.subgraph(males)
    female_network = network.subgraph(females)
    
    all_edges = set(network.edges())
    female_edges = set(female_network.edges())
    male_edges = set(male_network.edges())
    
    # Draw network in bipartite layout, removing edges between female agents or male agents
    edgelist = all_edges - female_edges - male_edges
    plt.subplot(131)
    plt.axis('off')
    pos = nx.bipartite_layout(network, females)
    nx.draw_networkx_nodes(G = network, nodelist = females, pos = pos, with_labels = False, node_color = 'r')
    nx.draw_networkx_nodes(G = network, nodelist = males, pos = pos, with_labels = False, node_color = 'b')
    nx.draw_networkx_edges(G = network, edgelist = edgelist, pos = pos)

    # Draw network of only male agents 
    plt.subplot(132)
    plt.axis('off')
    nx.draw_networkx(G = male_network, pos = nx.kamada_kawai_layout(male_network), \
                    with_labels = False, node_color = 'b')

    # Draw network of only female agents
    plt.subplot(133)
    plt.axis('off')
    nx.draw_networkx(G = female_network, pos = nx.kamada_kawai_layout(female_network), \
                     with_labels = False, node_color = 'r')

    plt.savefig('Class{}step{}.png'.format(class_id, step))