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

    m.params.TimeLimit = 15

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

def linear_solve(t,T):
    E = np.ones((len(t),len(t[0])))

    M = E.shape[0]
    N = E.shape[1]

    m = Model("mip")

    phi = m.addVars(M,N,vtype=GRB.CONTINUOUS)
    z = m.addVars(M,vtype=GRB.CONTINUOUS)
    z_t = m.addVar(vtype=GRB.CONTINUOUS)

    m.setObjective(z_t, GRB.MINIMIZE)


    for i in range(0,len(t)):
        for j in range(0,len(t[i])):
            if t[i][j] > T:
                m.addConstr(phi[i,j] <= 0)

    m.addConstrs((z_t >= z[i] for i in range(M)))
    m.addConstrs((z[i] == quicksum((phi[i,j])*t[i,j] for j in range(N)) for i in range(M)))
    m.addConstrs((quicksum(phi[i,j]*E[i][j] for i in range(M)) == 1) for j in range(N))

    m.optimize()

    opt_makespan = m.objVal
    
    var = []
    for i in range(0,len(t)):
        v = []
        for j in range(0,len(t[i])):
            v.append(phi[i,j].X)
        var.append(v)
    return opt_makespan,var

def mach(i):
    return "m"+str(i)

def op(j):
    return "n"+str(j)

def lenstra(m,n):    
    #t = np.random.randint(50,size=(m,n))+1
    t_ = np.random.randint(50,size=n)+1
    t = []
    for i in range(0,m):
        f = random.random()*0.5 + 0.5
        t.append(f*copy.deepcopy(t_))
    t = np.asarray(t)
    
    opt_makespan = MIP_solve(t)
    T_min = max(np.min(t,axis=0))
    T_max = 50
    
    for T in range(math.floor(T_min)+1,T_max+1):
        opt_m,phi = linear_solve(t,T)
        if opt_m <= T:
            break
    
    G = nx.Graph()
    for i in range(0,len(phi)):
        for j in range(0,len(phi[i])):
            if phi[i][j] < 1 and phi[i][j] > 0:
                G.add_node(mach(i))
                G.add_node(op(j))
                G.add_edge(mach(i),op(j))
                
    pairs = []
    while(len(G.edges) > 0):
        nodes = list(G.nodes)
        
        for i in range(0,len(nodes)):
            if G.degree(nodes[i]) == 1:
                break
        if G.degree(nodes[i]) == 0:
            for k in range(0,len(nodes)):
                if G.degree(nodes[k]) == 2:
                    i = k
                    break
        nbrs = list(G[nodes[i]])
        
        temp = 0
        if nodes[i][0] == 'n':
            pairs.append([nbrs[0],nodes[i]])
        else:
            pairs.append([nodes[i],nbrs[0]])
        G.remove_node(nodes[i])
        G.remove_node(nbrs[0])
    
    for i in range(0,len(pairs)):
        m_ind = int(pairs[i][0][1:])
        n_ind = int(pairs[i][1][1:])
        print(m_ind," ",n_ind)
        phi[m_ind][n_ind] = 1
    
    for i in range(0,len(phi)):
        for j in range(0,len(phi[i])):
            if phi[i][j] < 1:
                phi[i][j] = 0
    
    makespan = max(np.sum(phi*t,axis=1))
    return makespan/opt_makespan
    
m_vals = [2,5,10,20]
n_vals = [25,50,100,200,500]

record1_makespan = np.zeros((len(m_vals),len(n_vals)))

for i in range(0,len(m_vals)):
    for j in range(0,len(n_vals)):
        ratios = []
        m = m_vals[i]
        n = n_vals[j]
        for k in range(0,5):
            r = lenstra(m,n)
            ratios.append(r)
        record1_makespan[i][j] = np.mean(ratios)
        
#print(record1_makespan)

np.savetxt('lenstra_prop.csv',record1_makespan, delimiter=',')
   
    
    
    
    
    
    
    