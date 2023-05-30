# exhaustive evaluation
# This code runs the original SC model for given input points in file 'in_params.csv'
# Parallelization is not implemented here. This code will run sequentially. 
# (It is not recommended to run this. Code to attempt the same task with parallelization is included in the same folder.)
import SupplyChainModelv2 as model
import numpy as np
import pandas as pd
import os
from csv import writer

distributor1 = {'name':'D1',
                'S': 500, # inventory capacity
                's': 350, # inventory threshold
                'H': 1, # inventory holding cost
                'C': [500], # delivery cost from manufacturer
                'D': [7] # delivery time
               }

distributor2 = {'name':'D2',
                'S': 500,
                's': 350,
                'H': 1,
                'C': [500],
                'D': [8]}

retailer1 = {'name':"R1",
             'S': 500,
             's': 300,
             'H': 10,
             'C':[5000,6000], # delivery cost per distributor
             'D':[2,3] # delivery time per distributor
            } 
retailer2 = {'name':"R2",
             'S': 500,
             's': 300,
             'H': 10,
             'C':[7000,5500],
             'D':[3,2]}

R_list = [retailer1,retailer2]
D_list = [distributor1,distributor2]

# this fun sets the design parameters of the supply chain

def N_sim_runs(D_list,R_list,
               S_D1, s_D1, S_D2, s_D2, S_R1, s_R1, S_R2, s_R2,
               arr_rate,p,NUM_OF_DAYS,Profit,
               NUM_OF_SIMS):
    R_list[0]['S'] = S_R1
    R_list[0]['s'] = s_R1
    R_list[1]['S'] = S_R2
    R_list[1]['s'] = s_R2
    D_list[0]['S'] = S_D1
    D_list[0]['s'] = s_D1
    D_list[1]['S'] = S_D2
    D_list[1]['s'] = s_D2
    
    avg_stats = []
    avg_nstats = []
    for i in range(NUM_OF_SIMS):
        frac_cust_ret, avg_profit, avg_hold_c, avg_del_c, timed_avg_nitems, avg_net_profit, nwise_stats = model.single_sim_run(lam=arr_rate, D_list=D_list, R_list=R_list, p=p, NUM_OF_DAYS=NUM_OF_DAYS, P=Profit)
        avg_stats.append([frac_cust_ret, avg_profit, avg_hold_c, avg_del_c, timed_avg_nitems, avg_net_profit])
        for i in range(len(nwise_stats)):
            if(len(avg_nstats)<=i):
                nwise_stats[i].pop(0)
                avg_nstats.append([x/NUM_OF_SIMS for x in nwise_stats[i]])
            else:
                for j in range(1,len(nwise_stats[i])):
                    avg_nstats[i][j-1] = avg_nstats[i][j-1] + nwise_stats[i][j]/NUM_OF_SIMS

    avg_stats = np.array((avg_stats))
    avg_stats = np.mean(avg_stats,axis=0)
    temp = []
    for i in avg_nstats:
        for j in i:
            temp.append(j)
    return [S_R1,s_R1,S_R2,s_R2,S_D1,s_D1,S_D2,s_D2,*avg_stats,*temp]

# parameters
lambda_arr_rate = 20
p = [0.5,0.5]
Profit = 100
num_days = 90
num_sims = 60

in_param_pd = pd.read_csv('../data/in_params.csv')
in_param_arr = np.array((in_param_pd.values))

for i in in_param_arr:
    avg_stats = N_sim_runs(D_list=D_list,R_list=R_list,
               S_D1=i[0], s_D1=i[1], S_D2=i[2], s_D2=i[3], S_R1=i[4], s_R1=i[5], S_R2=i[6], s_R2=i[7],
               arr_rate=lambda_arr_rate,p=p,NUM_OF_DAYS=num_days,Profit=Profit,NUM_OF_SIMS=num_sims)
    with open('../data/supplychain_datav2_90_60.csv', 'a', newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow([lambda_arr_rate, Profit, num_days, num_sims, (p[0],p[1]), *avg_stats])
        f_object.close()
