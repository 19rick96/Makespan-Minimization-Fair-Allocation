import numpy as np
import math
import random
import time
import copy
import networkx as nx
import matplotlib.pyplot as plt
from gurobipy import *

M = 100000

def MIP_solve(t):
    E = np.ones((len(t),len(t[0])))

    M = E.shape[0]
    N = E.shape[1]

    m = Model("mip")

    phi = m.addVars(M,N,vtype=GRB.BINARY)
    z = m.addVars(M,vtype=GRB.CONTINUOUS)
    z_t = m.addVar(vtype=GRB.CONTINUOUS)

    m.setObjective(z_t, GRB.MINIMIZE)

    m.addConstrs((z_t >= z[i] for i in range(M)))
    m.addConstrs((z[i] == quicksum((phi[i,j])*t[i,j] for j in range(N)) for i in range(M)))
    m.addConstrs((quicksum(phi[i,j]*E[i][j] for i in range(M)) == 1) for j in range(N))

    m.optimize()

    opt_makespan = m.objVal

    return opt_makespan

def ag(i):
    return "a"+str(i)

def ch(i):
    return "c"+str(i)

def phase1_g(t):
    G = nx.DiGraph()
    for i in range(0,len(t)):
        G.add_node(ag(i),z=0)
    for i in range(0,len(t[0])):
        G.add_node(ch(i),price = 0)
    alloc = np.argmin(t,0) 
    for i in range(0,len(alloc)):
        G.add_edge(ag(alloc[i]),ch(i),weight = t[alloc[i]][i])
        G.nodes[ch(i)]['price'] = t[alloc[i]][i]
        for j in range(0,len(t)):
            if j != alloc[i] and t[j][i] == t[alloc[i]][i]:
                G.add_edge(ch(i),ag(j),weight=t[j][i])
    for i in range(0,len(t)):
        s = 0
        alloc_chores = list(G.adj[ag(i)])
        for j in range(0,len(alloc_chores)):
            s = s + G.nodes[alloc_chores[j]]['price']
        G.nodes[ag(i)]['z'] = s
    return G

def reference_agent(G,n):
    completion_times = []
    for i in range(0,n):
        agent = ag(i)
        alloc_chores = list(G.adj[agent])
        s = 0
        for j in range(0,len(alloc_chores)):
            s = s + G.edges[(agent,alloc_chores[j])]['weight']
        completion_times.append(s)
    i = np.argmax(completion_times)
    return i,max(completion_times)

def reachability_set_g(G,i):
    R = []
    R.append([ag(i)])
    l = 0
    while(len(R[l]) != 0):
        R.append([])
        for i in range(0,len(R[l])):
            alloc_chores = list(G.adj[R[l][i]])
            k = len(alloc_chores)-1
            while k >= 0:
                candidate_agents = list(G.adj[alloc_chores[k]])
                for j in range(0,len(candidate_agents)):
                    if any(candidate_agents[j] in sublist for sublist in R) != True:
                        R[l+1].append(candidate_agents[j])
                k = k-1
        l = l+1
    return R

def phase2_g(G,eps,n):
    i,makespan = reference_agent(G,n)
    R = reachability_set_g(G,i)
    l = 1
    while len(R[l]) != 0:
        flag = 0
        for j in range(0,len(R[l])):
            pred = list(G.predecessors(R[l][j]))
            for k in range(0,len(pred)):
                prev_owner = list(G.predecessors(pred[k]))[0]
                new_weight = G.edges[(pred[k],R[l][j])]['weight']
                old_weight = G.edges[(prev_owner,pred[k])]['weight']
                if ((1+eps)*(G.nodes[R[l][j]]['z'] + new_weight) < makespan) and (prev_owner in R[l-1]):
                    G.add_edge(R[l][j],pred[k],weight = new_weight)
                    G.nodes[R[l][j]]['z'] = G.nodes[R[l][j]]['z'] + new_weight
                    G.nodes[prev_owner]['z'] = G.nodes[prev_owner]['z'] - old_weight
                    G.add_edge(pred[k],prev_owner,weight = old_weight)
                    G.remove_edge(prev_owner,pred[k])
                    G.remove_edge(pred[k],R[l][j])
                    
                    i,makespan = reference_agent(G,n)
                    R = reachability_set_g(G,i)
                    l = 1
                    flag = 1
                    break
            if flag == 1:
                break
        if flag == 0:
            l = l+1
    return G,R

def phase3_g(G,t,R):
    unreachable_agents = []
    reachable_agents = []
    reachable_chores = []
    for i in range(0,len(t)):
        if any(ag(i) in sublist for sublist in R) != True:
            unreachable_agents.append(ag(i))
        else:
            reachable_agents.append(ag(i))

    for i in range(0,len(reachable_agents)):
        agent = reachable_agents[i]
        pred = list(G.predecessors(agent))
        for k in range(0,len(pred)):
            other_owner = list(G.predecessors(pred[k]))[0]
            if other_owner in unreachable_agents:
                G.remove_edge(pred[k],agent)
    
    for i in range(len(R)):
        for j in range(0,len(R[i])):
            chores = list(G.adj[R[i][j]])
            reachable_chores = reachable_chores + chores

    ratios = []
    for i in range(0,len(unreachable_agents)):
        ratio_arr = []
        for j in range(0,len(reachable_chores)):
            #print(list(G.adj[unreachable_agents[i]]))
            beta = 1
            if len(list(G.adj[unreachable_agents[i]])) != 0:
                adj_chore = list(G.adj[unreachable_agents[i]])[0]
                beta = G.edges[(unreachable_agents[i],adj_chore)]['weight']/G.nodes[adj_chore]['price']
            r = G.nodes[reachable_chores[j]]['price']
            w = t[int(unreachable_agents[i][1:])][int(reachable_chores[j][1:])]
            d = w/(beta*r)
            ratio_arr.append(d)
        ratios.append(ratio_arr)

    delta = np.min(ratios)

    for i in range(0,len(reachable_chores)):
        chore = reachable_chores[i]
        G.nodes[chore]['price'] = G.nodes[chore]['price']*delta
    for i in range(0,len(ratios)):
        for j in range(0,len(ratios[i])):
            if ratios[i][j] == delta:
                agent = unreachable_agents[i]
                chore = reachable_chores[j]
                G.add_edge(chore,agent,weight=t[int(agent[1:])][int(chore[1:])])
    return G

def check_full(R,n):
    reachable_agents = []
    for i in range(0,len(R)):
        reachable_agents = reachable_agents + R[i]
    if len(reachable_agents) == n:
        return True
    else:
        return False
    
def market_based_algo(t):
    eps = 1/(3*np.max(t)*len(t[0]))
    G = phase1_g(t)
    cnt = 0
    while(1):
        cnt += 1
        G,R = phase2_g(G,eps,len(t))
        if check_full(R,len(t)) == True:
            break
        G = phase3_g(G,t,R)
    i,makespan = reference_agent(G,len(t))
    return G,makespan,cnt

###############################################

def find_makespan(t_,G,T,parts):
    t = copy.deepcopy(t_)
    t = M*(t>T) + t

    m = len(t)
    n = len(t[0])

    G,makespan,num_compares = market_based_algo(t)

    return makespan,num_compares

def run_experiment(m,n,prob,pruning=False,id_flag="nonid"):
    t = np.random.randint(50,size=(m,n))+1
    if id_flag == "id":
        t_ = np.random.randint(50,size=n)+1
        #t_ = np.array([22 , 7 ,16, 20, 15, 26, 23, 25, 44, 47, 17, 10, 19,  9 , 2 ,20 ,24 , 8 ,31 ,23])
        t = []
        for i in range(0,m):
            t.append(copy.deepcopy(t_))
        t = np.asarray(t)
    elif id_flag == "prop":
        t_ = np.random.randint(50,size=n)+1
        t = []
        for i in range(0,m):
            f = random.random()*0.5 + 0.5
            t.append(f*copy.deepcopy(t_))
        t = np.asarray(t)

    G = nx.Graph()
    G.add_nodes_from(range(m))
    for i in range(0,m):
        for j in range(i+1,m):
            G.add_edge(i,j)

    for e in G.edges:
        r = random.random()
        if r < prob:
            G.remove_edge(*e)

    makespan = -1
    num_compares = -1

    if pruning == False:
        items = range(n)

        parts_pos = sorted(random.sample(items,m-1))

        parts = []

        parts.append(list(items[0:parts_pos[0]]))
        for i in range(1,len(parts_pos)):
            parts.append(list(items[parts_pos[i-1]:parts_pos[i]]))
        parts.append(list(items[parts_pos[-1]:]))

        makespan,num_compares = find_makespan(t,G,np.max(t),parts)

    else:
        items = range(n)

        parts_pos = sorted(random.sample(items,m-1))

        parts = []

        parts.append(list(items[0:parts_pos[0]]))
        for i in range(1,len(parts_pos)):
            parts.append(list(items[parts_pos[i-1]:parts_pos[i]]))
        parts.append(list(items[parts_pos[-1]:]))

        m_arr = []
        n_iters_arr = []
        T_min = max(np.min(t,axis=0))
        T_max = 50
        thresh_ = t.flatten()
        thresh = []
        """
        for i in range(0,len(thresh_)):
            if thresh_[i] >= T_min and thresh_[i] not in thresh:
                thresh.append(thresh_[i])
        thresh.sort()
        """
        for T in range(math.floor(T_min)+1,T_max+1):
            thresh.append(T)
        """
        i = int(len(thresh)/2)
        flag = 0
        while flag == 0:
            m_,n_iter = find_makespan(t,G,thresh[i],parts)
            if 
        """
        i = 0
        while i < len(thresh) :
            print(i)
            m_,n_iter = find_makespan(t,G,thresh[i],parts)
            m_arr.append(m_)
            n_iters_arr.append(n_iter)
            """
            if len(m_arr) > 2 and m_arr[-1] < M/2:
                if m_arr[-1] > m_arr[-2]:
                    break
            """
            i = i + 1

        ind = np.argmin(m_arr)
        makespan = m_arr[ind]
        num_compares = n_iters_arr[ind]

    mod = Model("mip")

    mod.params.TimeLimit = 15

    phi = mod.addVars(m,n,vtype=GRB.BINARY)
    z_t = mod.addVar(vtype=GRB.CONTINUOUS)

    mod.setObjective(z_t, GRB.MINIMIZE)

    mod.addConstrs((z_t >= quicksum((phi[i,j])*t[i,j] for j in range(n)) for i in range(m)))
    mod.addConstrs((quicksum(phi[i,j] for i in range(m)) == 1) for j in range(n))

    mod.optimize()

    opt_makespan = mod.objVal

    return makespan/opt_makespan,num_compares

##################################################################################

m_vals = [2,5,10,20]
n_vals = [25,50,100,200,500]

m_vals = [10,20]
n_vals = [200]

##################################################################################

record2_makespan = np.zeros((len(m_vals),len(n_vals)))
record2_iters = np.zeros((len(m_vals),len(n_vals)))

for i in range(0,len(m_vals)):
    for j in range(0,len(n_vals)):
        ratios = []
        compares = []
        m = m_vals[i]
        n = n_vals[j]
        for k in range(0,5):
            r,c = run_experiment(m,n,0.0,True,"prop")
            ratios.append(r)
            compares.append(c)
        record2_makespan[i][j] = np.mean(ratios)
        record2_iters[i][j] = np.mean(compares)

print(record2_makespan)
np.savetxt('MBA_prop_pr_makespan.csv',record2_makespan, delimiter=',')
#np.savetxt('MAB/id_iters.csv',record2_iters, delimiter=',')
