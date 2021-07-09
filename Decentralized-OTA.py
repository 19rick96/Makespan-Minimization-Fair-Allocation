import random
import numpy as np
import copy
import time
from gurobipy import *
import networkx as nx

M = 100000

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
            else:
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

def op_trading(agents,G):
    impr = 1
    cnt = 0
    while impr == 1:
        cnt += 1
        impr = 0
        for e in G.edges:
            impr_ = agents[e[0]].gen_deqx(agents[e[1]])
            if impr_ == 1:
                impr = 1
    return cnt

def op_tr(agents):
    impr = 1
    cnt = 0
    while impr == 1:
        cnt += 1
        impr = 0
        for i in range(0,len(agents)):
            for j in range(i+1,len(agents)):
                impr_ = agents[i].gen_deqx(agents[j])
                if impr_ == 1:
                    impr = 1
    if cnt == 1:
        return 0
    else:
        return 1
                
def dec_op_trading(agents,G):
    impr = 1
    cnt = 0
    while impr == 1:
        cnt += 1
        impr = 0
        r = np.random.rand(len(agents))
        send = []
        rec = []
        for i in range(0,len(agents)):
            rec.append([])
        for i in range(0,len(agents)):
            nbrs = list(G[i])
            m = r[i]
            m_ind = -1
            for j in range(0,len(nbrs)):
                if r[nbrs[j]] > m:
                    m = r[nbrs[j]]
                    m_ind = nbrs[j]
            send.append(m_ind)
            if m_ind != -1:
                rec[m_ind].append(i)
        q = []
        for i in range(0,len(send)):
            if send[i] == -1:
                q.append(i)
        while len(q) != 0:
            ag_arr = []
            ind = q.pop()
            ag_arr.append(agents[ind])
            for i in range(0,len(rec[ind])):
                ag_arr.append(agents[rec[ind][i]])
                q.append(rec[ind][i])
            if len(ag_arr) > 1:
                impr_ = op_tr(ag_arr)
                if impr_ == 1:
                    impr = 1
    return cnt

def find_makespan(t_,G,T,parts):
    t = copy.deepcopy(t_)
    t = M*(t>T) + t

    m = len(t)
    n = len(t[0])

    agents = []

    for i in range(0,m):
        a = agent(t[i])
        a.allocate(parts[i])
        agents.append(a)

    num_compares  = dec_op_trading(agents,G)
    makespan = calc_makespan(agents)

    return makespan,num_compares

def run_experiment_prop(t,prob):
    m = len(t)
    n = len(t[0])
    
    G = nx.Graph()
    G.add_nodes_from(range(m))
    
    
    for i in range(0,m-1):
        G.add_edge(i,i+1)
    """
    for i in range(0,m):
        for j in range(i+1,m):
            G.add_edge(i,j)
    """
    G_partial = copy.deepcopy(G)

    for e in G_partial.edges:
        r = random.random()
        if r < prob:
            r_edge = e
            G_partial.remove_edge(*r_edge)
            if nx.is_connected(G_partial) == False:
                G_partial.add_edge(*r_edge)
    
    #########################################################################
    
    makespan = -1
    num_compares = -1

    items = range(n)
    parts_pos = sorted(random.sample(items,m-1))
    parts = []
    parts.append(list(items[0:parts_pos[0]]))
    for i in range(1,len(parts_pos)):
        parts.append(list(items[parts_pos[i-1]:parts_pos[i]]))
    parts.append(list(items[parts_pos[-1]:]))
    
    t1 = time.time()
    makespan,num_compares = find_makespan(t,G_partial,np.max(t),parts)
    t2 = time.time()
    
    #print("Partially Connected_Network : ",makespan," - ",num_compares)
    
    #print("Best Possible Makespan : ",t_sum/s_sum)
    
    #print("###########################################################")
    
    #########################################################################
    
    return makespan,num_compares,t2-t1

"""
m_vals = [5,10,20]
n_vals = [25,50,100,200,500]

data = []
data_var = []

for m in m_vals:
    for n in n_vals:
        r_arr = [0.0,0.2,0.5,0.7]

        UB = [0,0,0,0]
        itrs = [0,0,0,0]
        times = [0,0,0,0]

        UB_var = [0,0,0,0]
        itrs_var = [0,0,0,0]
        times_var = [0,0,0,0]
        
        for i in range(0,len(r_arr)):
            r = r_arr[i]
            m_span = []
            itrs_arr = []
            times_arr = []
            for k in range(0,20):
                t_ = np.random.randint(10,50,size=n)
                t_sum = sum(t_)
                t = []
                f_arr = []
                for j in range(0,m):
                    f = random.random()*0.5 + 0.5
                    #f = 1.0
                    f_arr.append(f)
                    t.append(f*copy.deepcopy(t_))
                t = np.asarray(t)
                
                print(m," ",n," ",k," ",2," ##########################################################")

                mod = Model("mip")
                mod.params.TimeLimit = 2
                phi = mod.addVars(m,n,vtype=GRB.BINARY)
                z_t = mod.addVar(vtype=GRB.CONTINUOUS)
                mod.setObjective(z_t, GRB.MINIMIZE)
                mod.addConstrs((z_t >= quicksum((phi[i,j])*(t_[j]*f_arr[i]) for j in range(n)) for i in range(m)))
                #mod.addConstrs((z_t >= quicksum((phi[i,j])*(t[i][j]) for j in range(n)) for i in range(m)))
                mod.addConstrs((quicksum(phi[i,j] for i in range(m)) == 1) for j in range(n))
                t1 = time.time()
                mod.optimize()
                t2 = time.time()
                opt_makespan = mod.objVal
                

                mspan_0,it_0,t_0 = run_experiment_prop(t,r)
                m_span.append(mspan_0/opt_makespan)
                itrs_arr.append(it_0)
                times_arr.append(t_0)
            UB[i] = np.mean(np.asarray(m_span))
            UB_var[i] = np.std(np.asarray(m_span))
        
            itrs[i] = np.mean(np.asarray(itrs_arr))
            itrs_var[i] = np.std(np.asarray(itrs_arr))

            times[i] = np.mean(np.asarray(times_arr))
            times_var[i] = np.std(np.asarray(times_arr)) 
        
        data.append([m,n,UB[0],itrs[0],times[0],UB[1],itrs[1],times[1],UB[2],itrs[2],times[2],UB[3],itrs[3],times[3]])
        data_var.append([m,n,UB_var[0],itrs_var[0],times_var[0],UB_var[1],itrs_var[1],times_var[1],UB_var[2],itrs_var[2],times_var[2],UB_var[3],itrs_var[3],times_var[3]])

np.savetxt('dec_ota_speed_2.csv',data,delimiter=',')
np.savetxt('dec_ota_speed_2_std.csv',data_var,delimiter=',')


#########################################################################################################

m_vals = [5,10,20]
n_vals = [25,50,100,200,500]

data = []
data_var = []

for m in m_vals:
    for n in n_vals:
        r_arr = [0.0,0.2,0.5,0.7]

        UB = [0,0,0,0]
        itrs = [0,0,0,0]
        times = [0,0,0,0]

        UB_var = [0,0,0,0]
        itrs_var = [0,0,0,0]
        times_var = [0,0,0,0]
        
        for i in range(0,len(r_arr)):
            r = r_arr[i]
            m_span = []
            itrs_arr = []
            times_arr = []
            for k in range(0,20):
                t_ = np.random.randint(10,50,size=n)
                t_sum = sum(t_)
                t = []
                f_arr = []
                for j in range(0,m):
                    f = random.random()*0.2 + 0.8
                    #f = 1.0
                    f_arr.append(f)
                    t.append(f*copy.deepcopy(t_))
                t = np.asarray(t)
                
                print(m," ",n," ",k," ",5," ##########################################################")

                mod = Model("mip")
                mod.params.TimeLimit = 2
                phi = mod.addVars(m,n,vtype=GRB.BINARY)
                z_t = mod.addVar(vtype=GRB.CONTINUOUS)
                mod.setObjective(z_t, GRB.MINIMIZE)
                mod.addConstrs((z_t >= quicksum((phi[i,j])*(t_[j]*f_arr[i]) for j in range(n)) for i in range(m)))
                #mod.addConstrs((z_t >= quicksum((phi[i,j])*(t[i][j]) for j in range(n)) for i in range(m)))
                mod.addConstrs((quicksum(phi[i,j] for i in range(m)) == 1) for j in range(n))
                t1 = time.time()
                mod.optimize()
                t2 = time.time()
                opt_makespan = mod.objVal
                

                mspan_0,it_0,t_0 = run_experiment_prop(t,r)
                m_span.append(mspan_0/opt_makespan)
                itrs_arr.append(it_0)
                times_arr.append(t_0)
            UB[i] = np.mean(np.asarray(m_span))
            UB_var[i] = np.std(np.asarray(m_span))
        
            itrs[i] = np.mean(np.asarray(itrs_arr))
            itrs_var[i] = np.std(np.asarray(itrs_arr))

            times[i] = np.mean(np.asarray(times_arr))
            times_var[i] = np.std(np.asarray(times_arr)) 
        
        data.append([m,n,UB[0],itrs[0],times[0],UB[1],itrs[1],times[1],UB[2],itrs[2],times[2],UB[3],itrs[3],times[3]])
        data_var.append([m,n,UB_var[0],itrs_var[0],times_var[0],UB_var[1],itrs_var[1],times_var[1],UB_var[2],itrs_var[2],times_var[2],UB_var[3],itrs_var[3],times_var[3]])

np.savetxt('dec_ota_speed_5.csv',data,delimiter=',')
np.savetxt('dec_ota_speed_5_std.csv',data_var,delimiter=',')

##################################################################################################################################

m_vals = [5,10,20]
n_vals = [25,50,100,200,500]

data = []
data_var = []

for m in m_vals:
    for n in n_vals:
        r_arr = [0.0,0.2,0.5,0.7]

        UB = [0,0,0,0]
        itrs = [0,0,0,0]
        times = [0,0,0,0]

        UB_var = [0,0,0,0]
        itrs_var = [0,0,0,0]
        times_var = [0,0,0,0]
        
        for i in range(0,len(r_arr)):
            r = r_arr[i]
            m_span = []
            itrs_arr = []
            times_arr = []
            for k in range(0,20):
                t_ = np.random.randint(10,50,size=n)
                t_sum = sum(t_)
                t = []
                f_arr = []
                for j in range(0,m):
                    f = random.random()*0.1 + 0.9
                    #f = 1.0
                    f_arr.append(f)
                    t.append(f*copy.deepcopy(t_))
                t = np.asarray(t)
                
                print(m," ",n," ",k," ",10," ##########################################################")
                mod = Model("mip")
                mod.params.TimeLimit = 2
                phi = mod.addVars(m,n,vtype=GRB.BINARY)
                z_t = mod.addVar(vtype=GRB.CONTINUOUS)
                mod.setObjective(z_t, GRB.MINIMIZE)
                mod.addConstrs((z_t >= quicksum((phi[i,j])*(t_[j]*f_arr[i]) for j in range(n)) for i in range(m)))
                #mod.addConstrs((z_t >= quicksum((phi[i,j])*(t[i][j]) for j in range(n)) for i in range(m)))
                mod.addConstrs((quicksum(phi[i,j] for i in range(m)) == 1) for j in range(n))
                t1 = time.time()
                mod.optimize()
                t2 = time.time()
                opt_makespan = mod.objVal
                

                mspan_0,it_0,t_0 = run_experiment_prop(t,r)
                m_span.append(mspan_0/opt_makespan)
                itrs_arr.append(it_0)
                times_arr.append(t_0)
            UB[i] = np.mean(np.asarray(m_span))
            UB_var[i] = np.std(np.asarray(m_span))
        
            itrs[i] = np.mean(np.asarray(itrs_arr))
            itrs_var[i] = np.std(np.asarray(itrs_arr))

            times[i] = np.mean(np.asarray(times_arr))
            times_var[i] = np.std(np.asarray(times_arr)) 
        
        data.append([m,n,UB[0],itrs[0],times[0],UB[1],itrs[1],times[1],UB[2],itrs[2],times[2],UB[3],itrs[3],times[3]])
        data_var.append([m,n,UB_var[0],itrs_var[0],times_var[0],UB_var[1],itrs_var[1],times_var[1],UB_var[2],itrs_var[2],times_var[2],UB_var[3],itrs_var[3],times_var[3]])

np.savetxt('dec_ota_speed_10.csv',data,delimiter=',')
np.savetxt('dec_ota_speed_10_std.csv',data_var,delimiter=',')

#############################################################################################################################

m_vals = [5,10,20]
n_vals = [25,50,100,200,500]

data = []
data_var = []

for m in m_vals:
    for n in n_vals:
        r_arr = [0.0,0.2,0.5,0.7]
        r_arr = [0.0]

        UB = [0,0,0,0]
        itrs = [0,0,0,0]
        times = [0,0,0,0]

        UB_var = [0,0,0,0]
        itrs_var = [0,0,0,0]
        times_var = [0,0,0,0]
        
        for i in range(0,len(r_arr)):
            r = r_arr[i]
            m_span = []
            itrs_arr = []
            times_arr = []
            for k in range(0,20):
                t_ = np.random.randint(10,50,size=n)
                t_sum = sum(t_)
                t = []
                f_arr = []
                for j in range(0,m):
                    f = random.random()*0.125 + 0.875
                    #f = 1.0
                    f_arr.append(f)
                    t.append(f*copy.deepcopy(t_))
                t = np.asarray(t)
                
                print(m," ",n," ",k," ",10," ##########################################################")
                mod = Model("mip")
                mod.params.TimeLimit = 2
                phi = mod.addVars(m,n,vtype=GRB.BINARY)
                z_t = mod.addVar(vtype=GRB.CONTINUOUS)
                mod.setObjective(z_t, GRB.MINIMIZE)
                mod.addConstrs((z_t >= quicksum((phi[i,j])*(t_[j]*f_arr[i]) for j in range(n)) for i in range(m)))
                #mod.addConstrs((z_t >= quicksum((phi[i,j])*(t[i][j]) for j in range(n)) for i in range(m)))
                mod.addConstrs((quicksum(phi[i,j] for i in range(m)) == 1) for j in range(n))
                t1 = time.time()
                mod.optimize()
                t2 = time.time()
                opt_makespan = mod.objVal
                

                mspan_0,it_0,t_0 = run_experiment_prop(t,r)
                m_span.append(mspan_0/opt_makespan)
                itrs_arr.append(it_0)
                times_arr.append(t_0)
            UB[i] = np.mean(np.asarray(m_span))
            UB_var[i] = np.std(np.asarray(m_span))
        
            itrs[i] = np.mean(np.asarray(itrs_arr))
            itrs_var[i] = np.std(np.asarray(itrs_arr))

            times[i] = np.mean(np.asarray(times_arr))
            times_var[i] = np.std(np.asarray(times_arr)) 
        
        data.append([m,n,UB[0],itrs[0],times[0],UB[1],itrs[1],times[1],UB[2],itrs[2],times[2],UB[3],itrs[3],times[3]])
        data_var.append([m,n,UB_var[0],itrs_var[0],times_var[0],UB_var[1],itrs_var[1],times_var[1],UB_var[2],itrs_var[2],times_var[2],UB_var[3],itrs_var[3],times_var[3]])

np.savetxt('line_data/dec_ota_speed_8.csv',data,delimiter=',')
np.savetxt('line_data/dec_ota_speed_8_std.csv',data_var,delimiter=',')

"""
import matplotlib.pyplot as plt

x = [25,50,100,200,500]

y1 = [[1.38,1.24,1.14,1.08,1.05],[1.36,1.28,1.19,1.09,1.05],[1.35,1.18,1.28,1.11,1.07],[1.27,1.26,1.18,1.07,1.14]]
y2 = [[1.58,1.43,1.30,1.16,1.08],[1.58,1.48,1.29,1.16,1.07],[1.29,1.42,1.25,1.15,1.07],[1.38,1.52,1.24,1.18,1.07]]

plt.plot(x, y2[0], label='maximum speed = 2')
plt.plot(x, y2[1], label='maximum speed = 5',linestyle="--")
plt.plot(x, y2[2], label='maximum speed = 8',linestyle="-.")
plt.plot(x, y2[3], label='maximum speed = 10',linestyle=":")
plt.xlabel("No. of operations (m) ")
plt.ylabel("Approximation Factor")
plt.legend()
plt.show()

