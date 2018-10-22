import pandas as pd
import networkx as nx

DATA_PATH = '../Data/'

def LoadData2Network(class_id, wave_num):

    sex_df = pd.read_csv(DATA_PATH + "{}_sex.csv".format(class_id), skiprows=0)
    trust_df = pd.read_csv(DATA_PATH + "{}_trust_w{}.csv".format(class_id, wave_num), header = 0, index_col = 0).fillna(0)
    trust_df.columns = trust_df.index.values
    trust_network = nx.from_pandas_adjacency(trust_df, create_using=nx.DiGraph())
    
    # Set node attributes (sex)
    for _, row in sex_df.iterrows():
        studentID = row[0]
        sex = row[1]
        trust_network.node[studentID]['sex'] = 'male' if sex == 1 else 'female'
    
    return trust_network