import pandas as pd
import networkx as nx

DATA_PATH = '../Data/'

def LoadData2Network(class_id, wave_num):

    sex_df        = pd.read_csv(DATA_PATH + "{}_sex.csv".format(class_id), skiprows=0)
    trust_df = pd.read_csv(DATA_PATH + "{}_trust_w{}.csv".format(class_id, wave_num), index_col=0)
    trust = nx.from_numpy_matrix(trust_df.fillna(0).values, create_using=nx.DiGraph())
    label_mapping = {idx: int(val) for idx, val in enumerate(trust_df.columns)}
    trust_network = nx.relabel_nodes(trust, label_mapping)
    
    # Set node attributes (sex)
    for _, row in sex_df.iterrows():
        studentID = row[0]
        sex = row[1]
        trust_network.node[studentID]['sex'] = 'male' if sex == 1 else 'female'
    
    return trust_network