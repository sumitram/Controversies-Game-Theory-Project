import numpy as np
import pandas as pd
import itertools
import networkx as nx

PAYOFF_FUNCTION = {('cooperate, cooperate'): (0.5, 0.5),
                   ('cooperate, defect'): (-1, 1),
                   ('defect, cooperate'): (1, -1),
                   ('defect, defect'): (-0.5, -0.5)}

class StudentAgent():
    def __init__(self, model, id, gender, trust_agents, affective):
        """
        Create a new agent

        Args:
            id: unique identifier for the agent
            gender: female or male
            prob_trust: tendency to trust others
            trust_agents: a set of agent ids which the agent trust
            affective: a dictionary of affective score rated by agent keyed by id
            strategy: either 'cooperate' or 'defect'
        """

        self.model = model
        self.id = id
        self.gender = gender
        self.trust_agents = trust_agents
        self.prob_trust = len(trust_agents) / len(self.model.agents)
        self.affective = affective
        self.strategy = None

    def play(self, opponent):
        if self.gender == 'female':
            bonus_for_gender = self.model.bonus_f2f if self.gender == opponent.gender else 0
        else:
            bonus_for_gender = self.model.bonus_m2m if self.gender == opponent.gender else 0

        bonus_for_friends = self.model.bonus_for_friends if self.affective[opponent.id] == 2 else 0

        prob_cooperate = self.prob_trust + bonus_for_gender + bonus_for_friends \
                         if opponent not in self.trust_agents else 1

        return np.random.choice(['cooperate', 'defect'], [prob_cooperate, 1 - prob_cooperate])

    def state_update(self):
        self.prob_trust = len(self.trust_agents) / len(self.model.players)



class TrustModel():
    def __init__(self, trust_network, affective_matrix, corr_friend_trust,\
                 bonus_m2m, bonus_f2f, bonus_for_friends):
        """
        Args:
            trust_network: A networkx DiGraph. Edge (A, B) exists iff A trusts B
            affective_matrix: A pandas dataframe that has the affective score for all agents
            friends_network: A networkx DiGraph. Edge (A, B) exists iff A rated B 2 in affective matrix.
            affected: A dictionary recording the incoming edges for each node in the friends_network
            bonus_m2m: normalized assortativity coefficient (male, male) interpreted as probability
            bonus_f2f: normalized assortativity coefficient (female, female) interpreted as probability
            bonus_for_friends: Correlation Coefficient between friends and trust
        """

        self.trust_network = trust_network
        self.affective_matrix = affective_matrix
        self.friends_network = self.__friends_network_from_affective_matrix__()
        self.affected = {}
        for idx in trust_network.nodes():
            self.affected[idx] =len(self.friends_network.in_edges(idx))
        self.bonus_m2m = bonus_m2m
        self.bonus_f2f = bonus_f2f
        self.bonus_for_friends = corr_friend_trust
        self.players = []

        for id, attributes in self.trust_network.nodes(data = True):
            trust_agents = [n for n in trust_network.neighbors[id]]
            affective = {}
            for column_idx, value in affective_matrix[id]:
                affective[column_idx] = value
            self.players.append(StudentAgent(self, id, attributes['sex'], set(trust_agents), affective))

    def step(self):
        """
        At each time step all pairs of agent will interact and play prisoner's dilemma once
        """

        for a, b in itertools.combinations(self.players, 2):
            strategy = (a.play(a, b), b.play(b, a))
            a_payoff, b_payoff = PAYOFF_FUNCTION[strategy]
            a.affective[b] += a_payoff
            b.affective[a] += b_payoff

            if a.affective[b] > 2:
                a.affective[b] = 2
            if a.affective[b] < -2:
                a.affective[b] = -2

            if b.affective[a] > 2:
                b.affective[a] = 2
            if b.affective[a] < -2:
                b.affective[a] = -2

            self.affective_matrix[a][b] = a.affective[b]
            self.affective_matrix[b][a] = b.affective[a] 
            
            # Update friends network and affected dictionary
            self.friends_network = self.__friends_network_from_affective_matrix__()
            self.affected = {}
            for idx in self.trust_network.nodes():
                self.affected[idx] =len(self.friends_network.in_edges(idx))    

            # Update trust network accordingly
            if strategy == ('cooperate', 'cooperate'):
                if (a.affective[b.id] == 2) and (b not in a.trust_agents):
                    # If to_trust = 1, we create a new edge from a to b, otherweise unchanged
                    to_trust = np.random.choice([0, 1], [1 - self.bonus_for_friends, self.bonus_for_friends])
                    if to_trust == 1:
                        a.trust_agents.add(b.id)
                        self.trust_network.add(a.id, b.id)

                if (b.affective[a.id] == 2) and (a not in b.trust_agents):
                    # If to_trust = 1, we create a new edge from b to a, otherweise unchanged
                    to_trust = np.random.choice([0, 1], [1 - self.bonus_for_friends, self.bonus_for_friends])
                    if to_trust == 1:
                        b.trust_agents.add(a.id)
                        self.trust_network.add(b.id, a.id)

            if strategy == ('cooperate', 'defect') and self.trust_network.has_edge(a.id, b.id):
                self.trust_network.remove(a.id, b.id)

            if strategy == ('defect', 'cooperate') and self.trust_network.has_edge(b.id, a.id):
                self.trust_network.remove(b.id, a.id)

            # Update prob_trust of both agents
            a.state_update()
            b.state_update()
            
        return self.trust_network

    def __friends_network_from_affective_matrix__(self):
        """
        Auxilary function to transform the affective matrix to friendship network
        """

        temporary_matrix = self.affective_matrix
        temporary_matrix.replace(-2, 0)
        temporary_matrix.replace(-1, 0)
        temporary_matrix.replace(1, 0)

        friendship = nx.from_pandas_adjacency(temporary_matrix, create_using = nx.DiGraph())

        return friendship
