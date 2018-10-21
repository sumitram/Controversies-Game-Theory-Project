from model import *

def run(num_step, trust_network, affective_matrix, corr_friend_trust, bonus_m2m, bonus_f2f, bonus_for_friends):
    model = TrustModel(trust_network = trust_network, affective_matrix = affective_matrix, \
                       corr_friend_trust = corr_friend_trust, bonus_m2m = bonus_m2m, \
                       bonus_f2f = bonus_f2f, bonus_for_friends = bonus_for_friends)

    networks = []
    current_step = 0
    while (current_step < num_step):
        networks.append(model.step())
        current_step += 1

    return networks