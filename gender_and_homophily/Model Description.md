## **Gender and Homophily in Iterated Prisoner's Dilelma Game**

We have two types of agents based on gender in our model. Each agent has his/her own tendency to trust other people, denoted by **prob_trust**, which can be defined intuitively as **out_degree(agent_id)/number_agents**. Followed by this definition it implies that this probability will be recalculated at each time step based on the updated trust network. As it will be seen, this probability together with our other assumptions will determine the mixed strategy for each agent.

To better fit in our context, we use "express self" to mean "cooperate" and "refrain" to mean "defect". The goal of our agent is to maximize their "payoff". In our case, it would be to maximize the friendship score others have for her/him. The payoff matrix roughly looks like this

| A\B          | express self | refrain   |
|--------------|--------------|-----------|
| **express self** | 0.5,0.5      | -1,1      |
| **refrain**      | 1,-1         | -0.5,-0.5 |

**Interpretation for the matrix in the end, but since the interpretation seems a bit fussy to me, I am gonna stick to "cooperate" and "defect" in the following sections.*


### **Evolutionary dynamics**
Before we introduce the evolutionary dynamics used in the current model, there are a few notations in consistent with the variable name you find in the code:
**prob_trust** = tendency to trust another agent, a number between 0 and 1 and defined as out_degree/num_agents.  
**bonus_f2f** = tendency for a female agent to trust another female agent, 0 if she interacts with a male agent  
**bonus_m2m** = tendency for a male agent to trust another male agent, 0 if he interacts with a female agent  
**bonus_for_friends** = tendency for agents to trust their friends (i.e. friendship score = 2), 0 if friendship score < 2


At each time step, all agents will interact with all other agents. For each female agent, she will play "express self" if she trust the person initially, otherwise her playing strategy "express self" is of probability **(prob_trust + bonus_f2f + bonus_for_friends)**. For each male agent, we simply replace bonus_f2f with bonus_m2m. 

Potential rewiring happens after each sinle interaction and friendship score updating. Following are all possible cases:

**Case 1:** (Strategy A, Strategy B) = (Cooperate, Cooperate)\
 If friendship_score(A to B) reaches 2 and the edge (A, B) did not exist, create a new edge (A,B) with probability of correlation between friends and trust. The case for B to A is symmetric.

**Case 2:** (Strategy A, Strategy B) = (Defect, Defect)\
This case will have no rewiring happenning, since under our dynamics defined above one will never defect if she/he trusts the other.

**Case 3:** (Strategy A, Strategy B) = (Defect, Cooperate)\
If edge (B, A) exists in the initial trust network, dissolve it. 

**Case 4:** (Strategy A, Strategy B) = (Cooperate, Defect)\
Symmetric with Case 3.


\
[*] Well I don't know how to better phrase it and emphasize the zero-sum property in prisoner's dilemma game. My idea is everyone has some degree of inferent desire to express herself/himself but also afraid of being judged/criticized for having different opinons at the same time. In reality, sharing a secret with others is a strong proof of trust to the other person but can also lower the barrier for the other person to trust back as well. I have seen similar statements multiple times in a psychology blog I subscribed, so we can find relevant papers to back this setting up as well. In terms of the situation of (refrain, express self)/(express self, refrain), we can interpret the payoff 1 as "stay in the comfort zone and feel good about knowing someone's secret"...Hmmm I refrain myself from interpreting in a much more cheesy way, but this may not seem very natural so we may want to discuss this later. Another risk is if we use this payoff to actually update our friendship score it will probability go beyond the bound (-2, 2).
