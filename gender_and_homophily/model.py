import numpy as np
import pandas as pd
import itertools
import networkx as nx

PAYOFF_FUNCTION = {('tell a secret, tell a secret'): (1, 1),
                   ('tell a secret, refrain'): (-1, 0),
                   ('refrain, tell a secret'): (0, -1),
                   ('refrain, refrain'): (0, 0)}

class StudentAgent():
    def __init__(self, model, id, gender, trust_agents, affective):
        """
        Create a new agent

        Args:
            id: unique identifier for the agent
            gender: female or male
            prob_trust: tendency to trust others (irrespective of gender)
            trust_agents: a set of agent ids which the agent trust
            affective: a dictionary of affective score rated by each agent keyed by id
            strategy: either 'tell a secret' or 'refrain'
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

        return np.random.choice(['tell a secret', 'refrain'], [prob_cooperate, 1 - prob_cooperate])

    def state_update(self):
        self.prob_trust = len(self.trust_agents) / len(self.model.players)



class TrustModel():
    def __init__(self, trust_network, affective_matrix, \
                 bonus_m2m, bonus_f2f, bonus_for_friends):
        """
        Args:
            trust_network: A networkx DiGraph. Edge (A, B) exists iff A trusts B
            affective_matrix: A pandas dataframe that has the affective scores from all agents to all agents
            bonus_m2m: normalized assortativity coefficient (male, male) interpreted as probability
            bonus_f2f: normalized assortativity coefficient (female, female) interpreted as probability
            bonus_for_friends: Correlation coefficient between friends and trust
        """

        self.trust_network = trust_network
        self.affective_matrix = affective_matrix
        self.bonus_m2m = bonus_m2m
        self.bonus_f2f = bonus_f2f
        self.bonus_for_friends = bonus_for_friends
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

        ## Update affective_matrix according the payoff function
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

            ## Rewiring in the trust network
            # Case 1
            if strategy == ('tell a secret', 'tell a secret'):
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
            # Case 3
            if strategy == ('refrain', 'tell a secret') and self.trust_network.has_edge(b.id, a.id):
                self.trust_network.remove(b.id, a.id)
            # Case 4
            if strategy == ('tell a secret', 'refrain') and self.trust_network.has_edge(a.id, b.id):
                self.trust_network.remove(a.id, b.id)

            ## Update prob_trust of both agents
            a.state_update()
            b.state_update()
            
        return self.trust_network