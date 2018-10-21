from run import *

def visualize(network):
    females = [idx for idx, sex in nx.get_node_attributes(network, 'sex').items() if sex == 'female']
    males = [idx for idx, sex in nx.get_node_attributes(network, 'sex').items() if sex == 'male']

    # Draw network in bipartite layout, removing edges between female agents or male agents

    # Draw network of only male agents 
    male_network = network.subgraph(males)

    # Draw network of only female agents
    female_network = network.subgraph(females)
