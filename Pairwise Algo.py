import random
import numpy as np
import copy
import time
from gurobipy import *
import networkx as nx

M = 100000

def phase_1(t):
    x = []
    p = []
    for i in range(0,len(t)):
        x.append([])
    alloc = np.argmin(t,0)
    for i in range(0,len(alloc)):
        x[alloc[i]].append(i)
    return np.asarray(x)

class agent:
    def __init__(self,time_arr):
        self.op_times = np.asarray(time_arr)
        self.alloc = []
        self.time_arr = []

    def eval_bundle(self,alloc):
        time_arr = np.asarray([]).astype(int)
        if len(alloc) != 0:
            time_arr = self.op_times[np.asarray(alloc)]
        return time_arr

    def allocate(self,alloc):
        self.alloc = np.asarray(alloc).astype(int)
        self.time_arr = self.eval_bundle(self.alloc)

    def total_time(self):
        return sum(self.time_arr)

    def check_deqx(self,agent2):
        v12 = self.eval_bundle(agent2.alloc)
        v21 = agent2.eval_bundle(self.alloc)

        z_11 = sum(self.time_arr)
        z_22 = sum(agent2.time_arr)

        if z_11 + min(v12) >= z_22 and z_22 + min(v21) >= z_11:
            return 1
        else:
            return 0

    def gen_deq1(self,agent2):
        tot_alloc = np.concatenate((self.alloc,agent2.alloc))
        if len(tot_alloc) == 0:
            return -1
        a1_new_alloc = []
        a2_new_alloc = []
        ratio = []

        v11 = self.eval_bundle(tot_alloc)
        v21 = agent2.eval_bundle(tot_alloc)
        v12 = []
        v22 = []

        ratio = v11/v21
        ind = np.argsort(ratio)[::-1]

        v11 = v11[ind]
        v21 = v21[ind]

        a1_new_alloc = tot_alloc[ind]

        z1 = np.sum(v11)
        z2 = 0

        for i in range(0,len(a1_new_alloc)):
            if max(z1 - v11[i] , z2 + v21[i]) < max(z1,z2):
                z1 = z1 - v11[i]
                z2 = z2 + v21[i]
                op_trans = a1_new_alloc[0]
                a1_new_alloc = np.delete(a1_new_alloc,0)
                a2_new_alloc = np.append(a2_new_alloc,op_trans)
            else:
                break

        improvement = 0
        #print(max(sum(self.time_arr),sum(agent2.time_arr))," ",max(z1,z2))
        if int(max(sum(self.time_arr),sum(agent2.time_arr))) > int(max(z1,z2)):
            improvement = 1
            self.allocate(a1_new_alloc)
            agent2.allocate(a2_new_alloc)

        return improvement

    def gen_deqx(self,agent2):
        flag = 0
        improvement = 0

        while flag == 0:
            v12 = self.eval_bundle(agent2.alloc)
            v21 = agent2.eval_bundle(self.alloc)

            z1 = sum(self.time_arr)
            z2 = sum(agent2.time_arr)

            if z1 == 0 and z2 == 0:
                break

            if z1 > z2:
                w1 = z1 - self.time_arr
                w2 = z2 + v21
                z_new = np.maximum(w1,w2)
                ind = np.argmin(z_new)
                if z_new[ind] < max(z1,z2):
                    transfer_op = int(self.alloc[ind])
                    self.alloc = np.delete(self.alloc,ind)
                    self.time_arr = np.delete(self.time_arr,ind)
                    agent2.alloc = np.append(agent2.alloc,transfer_op)
                    agent2.time_arr = np.append(agent2.time_arr,v21[ind])
                    improvement = 1
                else:
                    flag = 1
            elif z1 < z2:
                w2 = z2 - agent2.time_arr
                w1 = z1 + v12
                z_new = np.maximum(w1,w2)
                ind = np.argmin(z_new)
                if z_new[ind] < max(z1,z2):
                    transfer_op = int(agent2.alloc[ind])
                    agent2.alloc = np.delete(agent2.alloc,ind)
                    agent2.time_arr = np.delete(agent2.time_arr,ind)
                    self.alloc = np.append(self.alloc,transfer_op)
                    self.time_arr = np.append(self.time_arr,v12[ind])
                    improvement = 1
                else:
                    flag = 1
        return improvement

def calc_makespan(agents):
    ct = []
    for i in range(0,len(agents)):
        ct.append(agents[i].total_time())
    return max(ct)

def pairwise_deq1(agents,G):
    impr = 1
    cnt = 0
    while impr == 1:
        cnt += 1
        impr = 0
        for e in G.edges:
            #print(cnt)
            impr_ = agents[e[0]].gen_deq1(agents[e[1]])
            #impr = agents[e[0]].gen_deqx(agents[e[1]])
            if impr_ == 1:
                impr = 1
    return cnt

def op_trading(agents,G):
    impr = 1
    cnt = 0
    while impr == 1:
        cnt += 1
        impr = 0
        for e in G.edges:
            #impr = agents[e[0]].gen_deq1(agents[e[1]])
            impr_ = agents[e[0]].gen_deqx(agents[e[1]])
            if impr_ == 1:
                impr = 1
    return cnt

def find_makespan(t_,G,T,parts,ota_flag):
    t = copy.deepcopy(t_)
    t = M*(t>T) + t

    m = len(t)
    n = len(t[0])

    agents = []

    for i in range(0,m):
        a = agent(t[i])
        a.allocate(parts[i])
        agents.append(a)

    if ota_flag == "ota":
        num_compares  = op_trading(agents,G)
    else:
        num_compares  = pairwise_deq1(agents,G)

    makespan = calc_makespan(agents)

    parts = []
    for i in range(0,len(agents)):
        parts.append(agents[i].alloc)

    return makespan,num_compares,parts

def run_experiment(m,n,prob,pruning=False,id_flag="nonid"):
    t = np.random.randint(50,size=(m,n))+1
    if id_flag == "id":
        t_ = np.random.randint(50,size=n)+1
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

        #makespan,num_compares,parts = find_makespan(t,G,np.max(t),parts,"mba")
        makespan,num_compares,parts = find_makespan(t,G,np.max(t),parts,"mba")

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
        thresh_ = t.flatten()
        thresh = []
        for i in range(0,len(thresh_)):
            if thresh_[i] >= T_min and thresh_[i] not in thresh:
                thresh.append(thresh_[i])
        thresh.sort()

        i = 0
        parts_arr = []
        while i < len(thresh) :
            #print(i)
            m_,n_iter,parts = find_makespan(t,G,thresh[i],parts,"mba")
            m_arr.append(m_)
            n_iters_arr.append(n_iter)
            parts_arr.append(copy.deepcopy(parts))
            """
            if len(m_arr) > 2 and m_arr[-1] < M/2:
                if m_arr[-1] > m_arr[-2]:
                    break
            """
            i = i + 1

        ind = np.argmin(m_arr)
        makespan = m_arr[ind]
        num_compares = n_iters_arr[ind]
        parts = parts_arr[ind]

        #makespan,num_compares,parts = find_makespan(t,G,np.max(t),parts,"ota")

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

##################################################################################

record1_makespan = np.zeros((len(m_vals),len(n_vals)))
record1_iters = np.zeros((len(m_vals),len(n_vals)))

for i in range(0,len(m_vals)):
    for j in range(0,len(n_vals)):
        ratios = []
        compares = []
        m = m_vals[i]
        n = n_vals[j]
        for k in range(0,5):
            r,c = run_experiment(m,n,0.0,False,"prop")
            ratios.append(r)
            compares.append(c)
        record1_makespan[i][j] = np.mean(ratios)
        record1_iters[i][j] = np.mean(compares)

np.savetxt('pairwiseMBA_pr_prop_makespan.csv',record1_makespan, delimiter=',')

##################################################################################
"""
record1_makespan = np.zeros((len(m_vals),len(n_vals)))
record1_iters = np.zeros((len(m_vals),len(n_vals)))

for i in range(0,len(m_vals)):
    for j in range(0,len(n_vals)):
        ratios = []
        compares = []
        m = m_vals[i]
        n = n_vals[j]
        for k in range(0,5):
            r,c = run_experiment(m,n,0.0,True,"nonid")
            ratios.append(r)
            compares.append(c)
        record1_makespan[i][j] = np.mean(ratios)
        record1_iters[i][j] = np.mean(compares)

np.savetxt('pairwiseMBA_prop_pr_makespan.csv',record1_makespan, delimiter=',')
"""
##################################################################################

