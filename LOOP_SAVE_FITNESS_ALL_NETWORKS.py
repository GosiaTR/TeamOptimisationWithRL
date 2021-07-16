import sys
import os
import random
import networkx as nx
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.colors as pltcolours

sys.path.append('/Users/gosiatatura/GOSIA_RESEARCH_ARL_2020/10_HUGO_CODE/')
from environment import Environment, get_action_num
import agents

############################################################################
def GENERATE_CONFIG(path_run,N,K,simulation_time,num_nodes,degree,history,learning_rate,episodes,epsilon_decay,GRAPH,min_max_name,path_networks):
    
    NAME="N_"+str(N)+"_K_"+str(K)+"_time_"+str(simulation_time) \
        +"_n_"+str(num_nodes)+"_d_"+str(degree)+"_memory_"+str(history)+"_lr"+str(learning_rate) \
        +"_ep_"+str(episodes)+"_dcy_"+str(epsilon_decay)  

    fNAME=path_run+NAME+".json"
    
    config={
        "title" : "xx",
        "deadline" : simulation_time,
        "episodes" : episodes,
        "95confidence": "true",
        "nk landscape" : {
            "N" : N,
            "K" : K
        },
        "graph": {
                "num nodes": num_nodes,
                "type": GRAPH,
                "degree": degree,
                "min_max_name": min_max_name,
                "path_networks": path_networks
                },
        "strategies" : [
            {
                "type" : "learnt",
                "name" : NAME,
                "episode" : "final",
                "config file" : fNAME,
                "alpha" : 1
            }
        ],
        "seed" : 24,
        "max processes" : 4
    }    
        
    return NAME,config

############################################################################
def EVALUATE_DYNAMICS(config):
    
    random.seed(config["seed"])
    np.random.seed(random.getrandbits(32))
    
    # generate graph
    graph_type = config["graph"]["type"]
    if graph_type == "regular":
        num_nodes = config["graph"]["num nodes"]
        degree = config["graph"]["degree"]
        graph = nx.circulant_graph(num_nodes, range(1,degree//2 +1))

    elif config["graph"]["type"] == "full":
        graph = nx.complete_graph(config["graph"]["num nodes"])

    elif config["graph"]["type"] == "min_max":
        num_nodes = config["graph"]["num nodes"]
        degree = config["graph"]["degree"] 
        min_max_name =  config["graph"]["min_max_name"]     
        
        path_read=config["graph"]["path_networks"] +"/DATA_social_networks_R/DATA_"+min_max_name+"/"
        path_loc=path_read+"/Degree_"+str(degree)+"/numNodes_"+str(num_nodes)+"/"
         
        SN_ITER=25
        INDEX=np.random.randint(1,SN_ITER)
        fNAME='Network_edge_list_iter_%d.txt'%(INDEX)
        graph=nx.read_edgelist(path_loc+fNAME, nodetype = int)

    # load strategies
    strategies = {}
    for strategy_cfg in config["strategies"]:
        strategy_type = strategy_cfg["type"]
        if strategy_type in ("learnt", "variable"):
            agent, agent_config = agents.from_config(
                    strategy_cfg["config file"],
                    get_action_num,
            )
            if agent_config["deadline"] != config["deadline"]:
                print("Warning: '" + strategy_cfg["name"] + \
                      "' has been set up for a different deadline.")
            if strategy_type == "learnt":
                if strategy_cfg["episode"]:
                    agent.load(suffix=strategy_cfg["episode"])
                else:
                    agent.load(suffix="final")
    
                agent_env_config = agent_config["training environment"]
                if agent_env_config["graph"] != config["graph"]:
                    print("Warning: '" + strategy_cfg["name"] + \
                          "' was trained on a different graph configuration.")
                if agent_env_config["nk landscape"] != config["nk landscape"]:
                    print("Warning: '" + strategy_cfg["name"] + \
                          "' was trained on a different",
                          "nk landscape configuration.")
    
            # add strategy to strategies dictionary
            strategies[strategy_cfg["name"]] = {
                "agent" : agent,
                "type" : strategy_cfg["type"],
                "alpha" : strategy_cfg["alpha"],
            }
    
    # the environment
    environment = Environment(
            config["nk landscape"]["N"],
            config["nk landscape"]["K"],
            graph,
            config["deadline"],
            max_processes=config["max processes"],
    )
    
    # fitnesses holds the mean fitness across all nodes at each time step
    # for each strategy and episode run.
    fitnesses = {}
    for strategy_name in strategies:
        fitnesses[strategy_name] = []
    
    for _ in range(config["episodes"]):
        environment.generate_new_fitness_func()
    
        for strategy_name, strategy_cfg in strategies.items():
            environment.reset()
    
            if strategy_cfg["type"] in ("learnt", "variable"):
                for time in range(config["deadline"]):
                    for node in range(config["graph"]["num nodes"]):
                        action = strategy_cfg["agent"].best_action(
                                node,
                                time,
                                environment,
                                )
                        environment.set_action(node, time, action)
    
                    environment.run_time_step(time)
    
            fitnesses[strategy_name].append(environment.get_mean_fitnesses())

    return fitnesses

############################################################################
path_hugo="/Users/gosiatatura/GOSIA_RESEARCH_ARL_2020/10_HUGO_CODE/"
path_exp="10_GOSIA_experiments_test/"

## configuration for regular network
#GRAPH="regular"
#num_nodes=100
#degree=10
#min_max_name='REG'

## configuration for fully connected network
GRAPH="full"
num_nodes=100
degree=num_nodes-1
min_max_name='FC'

## configurations for other networks
#GRAPH="min_max"
#num_nodes=100
#degree=10
#min_max_name='max_max_betweenness'

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

path_run=path_hugo+"experiments/"+path_exp+min_max_name+"/"
path_save=path_hugo+"experiments/"+path_exp+min_max_name+"_FITNESS/"
if not os.path.exists(path_save):
    os.mkdir(path_save) 

NAME,config=GENERATE_CONFIG(path_run,N,K,simulation_time,num_nodes,degree,history,learning_rate,episodes,epsilon_decay,GRAPH,min_max_name,path_hugo)
print(NAME)
fitnesses=EVALUATE_DYNAMICS(config)
# I know this loop is a very bad solution, but it was the simplest solution for me, since I didn't want to have to change 
# more code copied from compare.py. All those python data structures confuse me a little bit, so I decided on the path 
#of least effort, assuring correctness over elegance of code             
for strategy_name in fitnesses:    
    FITNESS=np.column_stack(fitnesses[strategy_name]) 

fNAME='FITNESS_ALL_'+NAME+'.txt'
np.savetxt(path_save+fNAME, FITNESS, delimiter='\t') 