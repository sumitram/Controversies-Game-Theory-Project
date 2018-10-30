import pandas as pd
import networkx as nx

DATA_PATH = '../Data/'

def load_data(class_id, wave_num):

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

def get_assortativity(network):
    assortativity_dict = nx.attribute_mixing_dict(network, 'sex', normalized = True)
    
    f2f = assortativity_dict['female']['female'] if 'female' in assortativity_dict['female']\
                                                               else 0
    f2f = round(f2f, 3)
    
    f2m = assortativity_dict['female']['male'] if 'male' in assortativity_dict['female']\
                                                             else 0
    f2m = round(f2m, 3)
    
    m2m = assortativity_dict['male']['male'] if 'male' in assortativity_dict['male']\
                                                           else 0
    m2m = round(m2m, 3)
    
    m2f = assortativity_dict['male']['female'] if 'female' in assortativity_dict['male']\
                                                             else 0
    m2f = round(m2f, 3)
    
    return (f2f, m2m, f2m, m2f)