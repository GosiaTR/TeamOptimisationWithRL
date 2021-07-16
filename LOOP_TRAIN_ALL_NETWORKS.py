import numpy as np
import time
import os

import subprocess
import glob
import shutil

import re
import json
import sys

def CreateJSON(NAME,simulation_time,learning_rate,discount_factor,history,epsilon_decay,episodes,N,K,num_nodes,degree,GRAPH,min_max_name,path_networks):
    Dict={ "name": NAME,
        "type": "QLearningAgent",
        
        "deadline": simulation_time,
        
        "learning rate": learning_rate,
        "discount factor": discount_factor,
        
        "state space": {
            "type": "time memory",
            "history": history
            },
        "exploration": {
            "type": "epsilon greedy",
            "epsilon start": 1,
            "epsilon decay": epsilon_decay
            },
        "possible actions": [
            "step",
            "best"
            ],
        "training environment": {
            "episodes": episodes,
            "save interval": episodes,
            "nk landscape": {
                "N": N,
                "K": K
                },
            "graph": {
                "num nodes": num_nodes,
                "type": GRAPH,
                "degree": degree,
                "min_max_name": min_max_name,
                "path_networks": path_networks
                },
            "seed": 349572,
            "max processes": 4
            }
        }
    return Dict

def SaveJSON(NAME,path_local,Dict):
    fNAME=path_local+NAME+".json"
    with open(fNAME,'w') as outfile:
        json.dump(Dict,outfile,indent=6)

def ExecuteJSON(NAME,path_hugo,min_max_name,path_exp):
    os.chdir(path_hugo)
    subprocess.call([" python train_all_networks.py experiments/"+path_exp+min_max_name+"/"+NAME+".json"],shell=True)

################################################################################################################
path_hugo="/Users/gosiatatura/GOSIA_RESEARCH_ARL_2020/10_HUGO_CODE/"
path_exp="10_GOSIA_experiments_test/"

## configuration for regular network
#GRAPH="regular"
#num_nodes=100
#degree=10
#min_max_name='REG'

## configuration for fully connected network
#GRAPH="full"
#num_nodes=100
#degree=num_nodes-1
#min_max_name='FC'

## configurations for other networks
GRAPH="min_max"
num_nodes=100
degree=10
min_max_name='max_max_betweenness'

#min_max_name=[ 'max_max_betweenness', 'max_mean_betweenness', \
#               'max_mean_clustering', 'min_max_closeness', \
#               'min_mean_betweenness', 'min_mean_clustering' ]



# landscape-simulation parameters
N=15
K=5
simulation_time=100

# ML parameters
discount_factor=1
history=2
learning_rate=0.001
episodes=1000
epsilon_decay=0.001

path_local=path_hugo+"/experiments/"+path_exp+min_max_name+"/"
if not os.path.exists(path_local):
    os.mkdir(path_local) 

NAME="N_"+str(N)+"_K_"+str(K)+"_time_"+str(simulation_time) \
    +"_n_"+str(num_nodes)+"_d_"+str(degree)+"_memory_"+str(history)+"_lr"+str(learning_rate) \
    +"_ep_"+str(episodes)+"_dcy_"+str(epsilon_decay)    

print(NAME)

Dict=CreateJSON(NAME,simulation_time,learning_rate,discount_factor,history,epsilon_decay,episodes,N,K,num_nodes,degree,GRAPH,min_max_name,path_hugo)
SaveJSON(NAME,path_local,Dict)
ExecuteJSON(NAME,path_hugo,min_max_name,path_exp)