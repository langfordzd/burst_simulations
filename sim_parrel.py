#!/home/zachary/condies/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 09:44:00 2023

@author: zachary
"""
# import os
# os.chdir('/home/zachary/projects/sim_amp_thresh/forgithub/')
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import numpy as np
import pandas as pd
import sim_mod as tm
import sim_parrel_plots as sp
import config as cfg
import random
import seaborn as sns
import itertools
import time
cm = sns.color_palette("rocket_r", as_cmap=True)
import help_funcs as hf
import ray
#%%
@ray.remote
def function(t_num, cycles,probabs,sims_per,wavelets,dists,max_rsq_is6):
    cfg.t_num = t_num
    cfg.thresholds = np.linspace(0.1,20,30)
    random.seed(100)
    r_seeds = random.sample(range(100, 1000), sims_per)
    result = []
    for d,distribution in enumerate(dists):  
        st = time.time()
        for probs in probabs:            
            dp_result = [tm.sim_data(rs,distribution,probs,cycles,wavelets,max_rsq_is6) for rs in r_seeds] 
            result.append(dp_result)
        et = time.time()
        print('Execution time:', et-st, 'seconds', d+1, 'out of', len(dists))    
    chars = pd.DataFrame(hf.flatten(result), columns=['ths', 'rsq', 'recov_fr', 'count', 'rec', 'falses', 
    'avg', 'reps', 'true_count','possible','quicks','prob','dist','snr2','nanavg',
    'raw_med', 'p', 'shape','scale','loc'])
    return chars
#%%
t_nums = 100#100
cfg.t_num = t_nums
cfg.thresholds = np.linspace(0.1,20,30)
sims_per = 10#10
probabs1 = [[0.1,0.9],[0.85,0.15],[0.97,0.03]] #[0.85,0.15],
probabs2 = [[0.1,0.9],[0.8,0.2]] #[0.85,0.15],
shape = list(np.round(np.linspace(0.1,1,9,dtype=float),3)) #9
scale =  list(np.round(np.geomspace(0.01,0.25,12,dtype=float),3)) #12
loc = list(np.round(np.linspace(0.001,0.05,3,dtype=float),3)) #3
temp = list(itertools.product(shape,scale,loc))
dists = tuple(temp)
#%%
ray.shutdown()
ray.init()
transient = function.remote(t_nums,[0.04,0.3],probabs1,sims_per,True,dists,False)
sustained_wave = function.remote(t_nums,[0.5,0.8],probabs2,sims_per,True,dists,False)
sustained_sine = function.remote(t_nums,[0.5,0.8],probabs2,sims_per,False,dists,False)
transient_is6 = function.remote(t_nums,[0.04,0.3],probabs1,sims_per,True,dists,True)
t, s_wave, s_sine, t_is6 = ray.get([transient, sustained_wave, sustained_sine, transient_is6])
ray.shutdown()
#%%
t['which'] = 't'
s_wave['which'] = 's_wave'
s_sine['which'] = 's_sine'
t_is6['which'] = 't_is6'
chars = pd.concat([t, s_wave, s_sine, t_is6], axis=0)
chars['falses'] = (chars['falses']+chars['reps']) #/ cfg.t_num
#chars['rec_false'] = chars['rec']/(chars['falses']+0.00001)
chars['rec'] = (chars['rec']) / chars['possible']
#chars['snrs'] = chars['snr2']
#%%
sp.fig1(chars,loc,scale,shape)
#%%
sp.fig2(chars,loc,scale,shape)
#%%
cut = chars[(chars['ths'] >= 5.5) & (chars['ths'] <= 6.5)].reset_index()
cut_hi_t = cut[(cut['p']==probabs1[0][1]) & (cut['which']=='t')]
cut_mi_t = cut[(cut['p']==probabs1[1][1]) & (cut['which']=='t')]
cut_lo_t = cut[(cut['p']==probabs1[2][1]) & (cut['which']=='t')]
cut_hi_s = cut[(cut['p']==probabs2[0][1]) & (cut['which']=='s_sine')]
cut_lo_s = cut[(cut['p']==probabs2[1][1]) & (cut['which']=='s_sine')]
#%%
sp.fig3(cut_hi_t,cut_mi_t,cut_lo_t,cut_hi_s,cut_lo_s)
