## **Gender and Homophily in Iterated Prisoner's Dilelma Game Model Description**

We have two types of agents based on gender in our model. Each agent has his/her own tendency to trust other people, denoted by **prob_trust**, which can be defined intuitively as **out_degree(agent_id)/number_agents**. Followed by this definition it implies that this probability will be recalculated at each time step upon trust network got updated. As it will be seen, this probability together with our other assumptions will determine the mixed strategy for each agent.

To better fit in our context, we use "express self" to mean "cooperate" and "refrain" to mean "defect". The goal of our agent is to gain trust from others (i.e. the incoming edges in the trust network). While this information is opaque to all agents, they know the strong correlation between frienships and trust, and therefore their interactions will change the affective scores, which will in turn, have an indirect affect on the trust network.

The payoff (affective score) matrix roughly looks like this

| A\B          | tell a secret | refrain   |
|--------------|---------------|-----------|
| **tell a secret** | 1,1      | -1,0      |
| **refrain**       | 0,-1         | 0,0 |


### **Evolutionary dynamics**
Before we introduce the evolutionary dynamics used in the current model, there are a few notations in consistent with the variable name you may encounter in the code:\
**prob_trust** = tendency to trust another agent, a number between 0 and 1 and defined as out_degree/num_agents.  
**bonus_f2f** = tendency for a female agent to trust another female agent, 0 if she interacts with a male agent  
**bonus_m2m** = tendency for a male agent to trust another male agent, 0 if he interacts with a female agent  
**bonus_for_friends** = tendency for agents to trust their friends (i.e. friendship score = 2), 0 if friendship score < 2


At each time step, all agents will interact with all other agents. For each female agent, she will play "express self" deterministically if she trust the person initially, otherwise the probability of her playing strategy "express self" is **(prob_trust + bonus_f2f + bonus_for_friends)**. For each male agent, we simply replace bonus_f2f with bonus_m2m. 

Potential rewiring happens after each sinle interaction and friendship score updating. Following are all possible cases:

**Case 1:** (Strategy A, Strategy B) = (Cooperate, Cooperate)\
 If friendship_score(A to B) reaches 2 and the edge (A, B) did not exist, create a new edge (A,B) with probability of correlation between friends and trust. The case for B to A is symmetric.

**Case 2:** (Strategy A, Strategy B) = (Defect, Defect)\
This case will have no rewiring happenning, since under our dynamics defined above one will never defect if she/he trusts the other.

**Case 3:** (Strategy A, Strategy B) = (Defect, Cooperate)\
If edge (B, A) exists in the initial trust network, dissolve it. 

**Case 4:** (Strategy A, Strategy B) = (Cooperate, Defect)\
Symmetric with Case 3.
