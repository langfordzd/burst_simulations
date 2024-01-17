#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 09:45:08 2023

@author: zachary
"""
def sim_data(r_seed,distribution,probs,cycles,wavelets,max_rsq_is6):

    import numpy as np
    import random
    from neurodsp.sim import sim_powerlaw,sim_cycle
    from neurodsp.filt import filter_signal
    from scipy.stats import lognorm
    from mne.time_frequency import tfr_array_morlet
    from scipy import signal as scisig
    import pandas as pd
    import config as cfg
    from scipy.stats import truncnorm
    
    random.seed(r_seed)
    np.random.seed(r_seed)
    
    def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
    
    amps_num = 5000
    outlier  = 3
    amp_dist = lognorm.rvs(distribution[0],loc=distribution[2],scale=distribution[1],size=amps_num,random_state=r_seed)
    amp_dist = amp_dist[amp_dist<(np.mean(amp_dist)+outlier*np.std(amp_dist))]
    amp_dist = amp_dist[:2000]
    
    cyc_dist = lognorm.rvs(0.5,loc=0.04,scale=0.12,size=amps_num,random_state=r_seed)
    cyc_dist = cyc_dist[cyc_dist<0.4]
    cyc_dist = cyc_dist[:2000]

    trun_norm = get_truncated_normal(mean=21, sd=1, low=15, upp=29)

    sfreq = cfg.sfreq
    trial_points = cfg.trial_points
    iti_points = cfg.iti_points
    trial_and_iti_points = trial_points+2*iti_points
    trial_chars = []
    sig_length = trial_and_iti_points * cfg.t_num
    sig_length_sec = sig_length/sfreq
    bot = np.arange(iti_points, sig_length, trial_points+2*iti_points)
    top = bot+cfg.trial_points
    signal = np.zeros(sig_length)
    sig_on_off = np.zeros(sig_length)

    population = [False,True]
    burst_last = False
    current_point = 0

    while current_point < sig_length-cfg.iti_points: 
        if random.choices(population,probs)[0] and not burst_last:
 
            cyc_freq = float(trun_norm.rvs(1))
            osc_amp =  np.random.choice(amp_dist)  
            if cycles[1] < 0.4:
                n_sec =  np.random.choice(cyc_dist)  
            else:
                n_sec = np.random.uniform(cycles[0], cycles[1])
                
            cycs = n_sec*cyc_freq
            
            if wavelets:
                s = cycs*sfreq / (2*cyc_freq*np.pi)
                M = int(cycs*sfreq/cyc_freq)
                w1 = np.real(scisig.morlet2(M, s, cycs))
                ratio = osc_amp/max(w1)
                signal_to_embed = w1*ratio
            else:                      
                cycs = n_sec*cyc_freq
                n_sam = int(np.floor(n_sec*sfreq))
                cyc_s = (sfreq / cyc_freq) / sfreq 
                rdsym = np.random.uniform(0.25,0.75)
                cyc = sim_cycle(cyc_s, sfreq, cycle_type='asine', rdsym = rdsym)
                signal_to_embed = osc_amp*np.tile(cyc, int(np.ceil(cycs)))[:n_sam]
                
            amp_mod_signal = (signal_to_embed)*random.choice([-1,1])
            in_bounds = np.arange(current_point,amp_mod_signal.size + current_point)  
            
            for trial, (a,b) in enumerate(zip(bot,top)): 
                if np.all((in_bounds > a) & (b > in_bounds)): 
                    trial_chars.append([current_point, current_point + amp_mod_signal.size, 
                                        current_point-a+cfg.iti_points,current_point+amp_mod_signal.size-a+cfg.iti_points,
                                        cyc_freq, cycs, n_sec, osc_amp, trial])
                    signal[in_bounds] = amp_mod_signal
                    sig_on_off[in_bounds] = 1
                    burst_last = True  
                    current_point = current_point+amp_mod_signal.size
                    break
    
        else:
            burst_last = False
            cycs = np.random.uniform(2,9)
            cyc_freq = np.random.uniform(15,29)
            current_point = current_point+int((cycs/cyc_freq) * sfreq)
        
    pl = sim_powerlaw(sig_length_sec, sfreq, exponent=-2)  
    signal = signal[0:sig_length]    
    signal_noise = np.sum([signal, pl], axis=0)
    signal_noise_big = np.pad(signal_noise, (5000, 5000), 'constant', constant_values=(0, 0))   
    signal_noise = np.nan_to_num(filter_signal(signal_noise_big, fs=sfreq, pass_type='bandpass',
                        filter_type='fir', f_range=(0.5, 50)))[5000:-5000]   #0.5
    bot = np.arange(iti_points, sig_length, trial_points+2*iti_points)
    top = bot+cfg.trial_points
    inter_ind = pd.IntervalIndex.from_arrays(bot, top, closed='both')
    
    if len(trial_chars) == 0:
        chars = pd.DataFrame(np.nan,index=[0], columns=['start', 'end', 'trial_start','trial_end','freq', 'n_cycs', 
                                                        'n_secs', 'amp','trial_number','trial'])
    else:
        chars = pd.DataFrame(trial_chars) 
        chars.columns =['start', 'end','trial_start','trial_end', 'freq', 'n_cycs', 'n_secs', \
                    'amp','trial_number']    
        chars['trial'] = pd.cut(chars['start'],bins=inter_ind) 

    cfg.chars = chars
    freqs = np.arange(12, 32.1, 0.5)
    cfg.freqs2 = freqs
    cfg.bfreqs = list(range(6,34+1,1))

    trials = np.reshape(signal_noise, (cfg.t_num, 1, trial_and_iti_points))
    tfr = np.squeeze(tfr_array_morlet(trials, cfg.sfreq, freqs, n_cycles=cfg.n_cycles, 
                                      output='power', use_fft=False, 
                                      zero_mean=True, n_jobs=20, 
                                      verbose=False))
    
    
    s_ = np.mean(tfr[:,6:34,200:325]) #this might work as well
    n_ = np.mean(tfr[:,6:34,0:125])
    snr2 = np.nanmean(s_/(n_))    
    cfg.snrs=snr2 
    raw_med = np.median(tfr)

    for f in range(0,tfr.shape[1]):
        freq = np.squeeze(tfr[:,f,:])
        med = np.median(freq[:,cfg.keeps], axis=(0,1))
        tfr[:,f,:] =  freq / med   

    trials = np.squeeze(trials[:,:,:])
    cfg.trials = trials
 
    
    tc = chars['trial_number'].value_counts()
    tc = tc.reset_index()
    uniq = tc['index'].unique()
    s = np.arange(0,cfg.t_num) 
    #we have to count the trials with zero bursts
    missing = [i for i in s if i not in uniq]
    d = pd.DataFrame(np.zeros((len(missing), 2)),columns=['index','trial_number'])
    tc = tc.append(d)
    true_count = tc['trial_number']  
    
    th, thresholds, rsq, t_num, recov_fr, count, recovered, falses, avg, repeat, sn, possible, quicks, nanavg = t_threshold(tfr,max_rsq_is6)

 
    return [
            th, 
            rsq, 
            recov_fr, 
            count, 
            recovered, 
            falses, 
            avg, 
            repeat,
            true_count,
            possible,
            quicks,
            probs,
            distribution,
            snr2,
            nanavg,
            raw_med,
            probs[1],
            distribution[0],
            distribution[1],
            distribution[2]
            ]#]
#%%
def t_threshold(t,max_rsq_is6):
    
    import numpy as np
    from scipy import stats
    import config as cfg
    import warnings
    from scipy.signal import peak_widths
    from skimage import feature as ff
    warnings.filterwarnings("ignore")
    sn = np.nanmean(cfg.snrs)
    t_num = cfg.t_num
    t_small = t[:,min(cfg.bfreqs):max(cfg.bfreqs),min(cfg.keeps):max(cfg.keeps)]
    tsm = np.mean(t_small, axis=(1,2))  
    rsq = [stats.pearsonr(np.sum((t_small > thresh)*1, axis=(1,2)),tsm)[0] for thresh in cfg.thresholds]
    if max_rsq_is6:
        max_rsq_thresh = 6
    else:
        max_rsq_thresh = cfg.thresholds[np.nanargmax(rsq)]
    
    count = np.zeros((t_num))   
    rec = []   
    for trial, tDat in enumerate(t):
        locs = ff.peak_local_max(tDat, min_distance=2, threshold_abs=max_rsq_thresh, threshold_rel=None, exclude_border=False)
        in_range = [locs[i][1] in cfg.keeps and locs[i][0] in cfg.bfreqs for i in range(len(locs))]
        max_idx = locs[in_range]
        for idx in max_idx:
            maxim = tDat[idx[0],idx[1]]
            atFreq = tDat[idx[0],:]
            hm = peak_widths(atFreq, [idx[1]], rel_height=0.5)
            length = (hm[0][0]/cfg.sfreq).astype(float)
            start = np.floor(hm[2][0]).astype(int)
            end = np.floor(hm[3][0]).astype(int)
            atAmps = tDat[:,idx[1]]
            #peakFreq = cfg.freqs2[idx[0]]
            hm = peak_widths(atAmps, [idx[0]], rel_height=0.5)
            low_freq = np.floor(hm[2][0]).astype(int)
            high_freq = np.floor(hm[3][0]).astype(int)
            spanFreq = cfg.freqs2[high_freq] - cfg.freqs2[low_freq]
            count[trial] =  count[trial]+1
            rec.append([trial,start,end, maxim, spanFreq, length])
    
    recovered, possible, falses, avg, repeat, quicks, nanavg = which_recovered(rec)
    if possible == 0:
        possible = 1
    rs = [r[3:6] for r in rec]
    
    return max_rsq_thresh, cfg.thresholds, rsq, t_num, rs, count, recovered, falses, avg, repeat, sn, possible, quicks, nanavg
#%%
def which_recovered(recov):
    import pandas as pd
    import numpy as np
    import config as cfg

    burst_cutoff = 0.3
    trials = cfg.trials
    bursts = np.zeros((len(recov),1000))
    bursts[:] = np.nan
    
    if len(recov) > 0:
        
        temp = pd.DataFrame(recov).iloc[:,[5]]<burst_cutoff
        quicks = temp.values.sum()
        
        recov = pd.DataFrame(recov).iloc[:,[0,1,2]]
        recov.columns =['trial_number','start', 'end']
        recov.dropna(inplace=True)
        false_burst = []
        hits = []
        repeat = []
        
        cfg.chars['recov_yet'] = False
        for r_,r in recov.iterrows():
            #
            waveform = trials[r['trial_number'],r['start']:r['end']]
            waveform = (waveform - np.mean(waveform))
            loc = np.argmax(abs(waveform))
            if waveform[loc] < 0:
                waveform = waveform*-1
            lt = len(waveform)
            mid = 500
            bursts[r_,mid-loc:mid+lt-loc] = waveform
            #
            fa = True
            t_chars = cfg.chars[cfg.chars['trial_number']==r['trial_number']]

            if not t_chars.empty:
                for t_, t in t_chars.iterrows():
                    estpoints = np.arange(r['start'],r['end'])
                    truepoints = np.arange(t['trial_start'], t['trial_end']) 
                    overlaps = np.intersect1d(estpoints,truepoints)
                    if overlaps.size/truepoints.size>0.0 and estpoints.size < truepoints.size*2: 
                        if not t['recov_yet']:
                            fa = False
                            hits.append(t_)    
                            cfg.chars.at[t_,'recov_yet'] = True
                        else:
                            repeat.append(1)
                            fa = False
                if fa:
                    false_burst.append(1)
            else:
                false_burst.append(1)
                        
        recovered = len(hits)
        falses = np.sum(false_burst)
        possible = len(cfg.chars)
        repeat = np.sum(repeat)

    else:
        recovered = 0
        falses = 0
        possible = len(cfg.chars)
        repeat = 0
        quicks = 0

    nanavg = np.nanmean(bursts,0)[425:575]
    bursts[np.isnan(bursts)] = 0
    avg = np.nanmean(bursts,0)[425:575]
   
    return recovered, possible, falses, avg, repeat, quicks,nanavg# true_count#, chars_['n_secs'].mean(), chars_['amp'].mean()
#%%
def flatten(l):
    return [item for sublist in l for item in sublist]
