#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:13:04 2023

@author: zachary
"""
#%%
def get_app_data(which):
    import pandas as pd
    import numpy as np
    r = which[['recov_fr','index']].reset_index()
    r_temp = pd.DataFrame(r['recov_fr'].explode())
    recov = pd.DataFrame(r_temp.recov_fr.tolist(), index= r_temp.index).rename(columns={0: "pow", 1: "freq", 2: "dur"}).reset_index()
    recov['bins_pow'] = pd.cut(recov['pow'], pd.IntervalIndex.from_breaks(np.linspace(5.5, 50, 50))).cat.codes + 5.5
    recov['bins_freq'] = pd.cut(recov['freq'], pd.IntervalIndex.from_breaks(np.linspace(0, 20, 20))).cat.codes #+ 12
    recov['bins_dur'] = pd.cut(recov['dur'], pd.IntervalIndex.from_breaks(np.linspace(0, 1, 20))).cat.codes

    xp = recov.groupby(['index','bins_pow'])['pow'].count().reset_index()
    c = xp.groupby(['index'])['pow'].sum().reset_index()
    xp = c.merge(xp, how = 'inner', on = ['index'])
    xp['pow'] = xp['pow_y']/xp['pow_x']
    yp = pd.DataFrame(xp.groupby(['bins_pow'])['pow'].mean())
    
    
    xf = recov.groupby(['index','bins_freq'])['freq'].count().reset_index()
    c = xf.groupby(['index'])['freq'].sum().reset_index()
    xf = c.merge(xf, how = 'inner', on = ['index'])
    xf['freq'] = xf['freq_y']/xf['freq_x']
    yf = pd.DataFrame(xf.groupby(['bins_freq'])['freq'].mean())
    
    xd = recov.groupby(['index','bins_dur'])['dur'].count().reset_index()
    c = xd.groupby(['index'])['dur'].sum().reset_index()
    xd = c.merge(xd, how = 'inner', on = ['index'])
    xd['dur'] = xd['dur_y']/xd['dur_x']
    yd = pd.DataFrame(xd.groupby(['bins_dur'])['dur'].mean())
  
    return xp,yp,xf,yf,xd,yd#
#%%
def get_test(which):
    import pandas as pd
    import numpy as np

    r = which[['recov_fr','index']].reset_index()
    r_temp = pd.DataFrame(r['recov_fr'].explode())
    recov = pd.DataFrame(r_temp.recov_fr.tolist(), index= r_temp.index).rename(columns={0: "pow", 1: "freq", 2: "dur"}).reset_index()
    p = recov[['index','pow']]
    p = [np.array(d['pow']) for _, d in p.groupby('index')]
    f = recov[['index','freq']]
    f = [np.array(d['freq']) for _, d in f.groupby('index')]
    d_ = recov[['index','dur']]
    d_ = [np.array(d['dur']) for _, d in d_.groupby('index')]

    return [p,f,d_]
#%%
def rotate_point(x, y, angle_rad):
    import numpy as np
    cos,sin = np.cos(angle_rad),np.sin(angle_rad)
    return cos*x-sin*y,sin*x+cos*y
#%%
def draw_brace(ax, span, position, text, text_pos, brace_scale=1.0, beta_scale=300., rotate=False, rotate_text=False):
    '''
        all positions and sizes are in axes units
        span: size of the curl
        position: placement of the tip of the curl
        text: label to place somewhere
        text_pos: position for the label
        beta_scale: scaling for the curl, higher makes a smaller radius
        rotate: true rotates to place the curl vertically
        rotate_text: true rotates the text vertically      
        
    '''
    import numpy as np
    # get the total width to help scale the figure
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin
    resolution = int(span/xax_span*100)*2+1 # guaranteed uneven
    beta = beta_scale/xax_span # the higher this is, the smaller the radius
    # center the shape at (0, 0)
    x = np.linspace(-span/2., span/2., resolution)
    # calculate the shape
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    # put the tip of the curl at (0, 0)
    max_y = np.max(y)    
    min_y = np.min(y)
    y /= (max_y-min_y)
    y *= brace_scale
    y -= max_y
    # rotate the trace before shifting
    if rotate:
        x,y = rotate_point(x, y, np.pi/2)
    # shift to the user's spot   
    x += position[0]        
    y += position[1]
    ax.autoscale(False)
    ax.plot(x, y, color='black', lw=1.1, clip_on=False)
    # put the text
    ax.text(text_pos[0], text_pos[1], text, fontsize=13, ha='center', va='bottom', rotation=90 if rotate_text else 0) 
#%%
def fig1(chars,loc,scale,shape):

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle
    import seaborn as sns
    fig = plt.figure(1, figsize=(12, 12))
    gs = gridspec.GridSpec(12, 24)
    gs.update(wspace=0.3, hspace=-.25)
    ys = [[0,2],[2,4],[4,6],[6,8],[8,10],[10,12]]
    which = 'loc'

    ts = chars[chars['which']=='t']
    l = loc[1]
    tloc = ts[ts['loc']==l]
    cg = tloc.groupby(['p'])
    for ii,(which,g_) in enumerate(cg):
        
        th = g_.groupby(['shape','scale'])['ths'].mean().reset_index()
        th['ths'] = th['ths'] - 6    
        thp = th.pivot('shape', 'scale', 'ths')
        thpt = thp.T
        mids = np.argwhere((np.array(thpt)>-0.6) & (np.array(thpt)<0.6))

        r = g_.groupby(['shape','scale'])['rec'].mean().reset_index()
        rp = r.pivot('shape', 'scale', 'rec')
        xtr2 = fig.add_subplot(gs[ys[ii][0]:ys[ii][1], 0:6])
        xtr2 = sns.heatmap(rp, cmap="rocket_r",cbar = True, vmin=0,vmax=1,
                    cbar_kws = dict(use_gridspec=False,location="top",ticks=[-0,1],shrink=0.6))
        for m in mids:
            xtr2.add_patch(Rectangle(m,0.9,0.9, fill=False, edgecolor='grey', lw=2, facecolor='grey', alpha=0.5,))
        xtr2.set_ylabel('')    
        xtr2.set_xlabel('') 
        if ii ==0:
            xtr2.annotate('A', xy=(3, 3), xytext=(-1.5, 1.5),weight='bold',fontsize=18)

        f = g_.groupby(['shape','scale'])['falses'].mean().reset_index()
        fp = f.pivot('shape', 'scale', 'falses')
        xtr3 = fig.add_subplot(gs[ys[ii][0]:ys[ii][1], 6:12])
        xtr3 = sns.heatmap(fp, cmap="rocket_r", cbar=True,vmin=0,vmax=100,
                           cbar_kws = dict(use_gridspec=False,location="top",ticks=[0,100],shrink=0.6))
        for m in mids:
            xtr3.add_patch(Rectangle(m,0.9,0.9, fill=False, edgecolor='grey', lw=2, facecolor='grey', alpha=0.5,))
        xtr3.set_ylabel('')    
        xtr3.set_xlabel('') 

        xtr1 = fig.add_subplot(gs[ys[ii][0]:ys[ii][1], 12:18])
        xtr1 = sns.heatmap(thp, cmap="vlag",cbar = True, vmin=-6,vmax=6,
                    #xticklabels=False, yticklabels=False,
                    cbar_kws = dict(use_gridspec=False,location="top",ticks=[-6,6],shrink=0.6))
        c_bar = xtr1.collections[0].colorbar
        c_bar.set_ticks([-6, 0, 6])
        c_bar.set_ticklabels(['0', '6', '>12'])
        xtr1.set_ylabel('')    
        xtr1.set_xlabel('')  

        c = g_.groupby(['shape','scale'])['snr2'].mean().reset_index()
        cp = c.pivot('shape','scale', 'snr2')
        xtr4 = fig.add_subplot(gs[ys[ii][0]:ys[ii][1], 18:24])
        xtr4 = sns.heatmap(cp, cmap="rocket_r", cbar=True,vmin=1,vmax=5, mask=cp.isnull(),
                           cbar_kws = dict(use_gridspec=False,location="top",shrink=0.6))
        c_bar = xtr4.collections[0].colorbar
        c_bar.set_ticks([1, 5])
        c_bar.set_ticklabels(['1', '5'])
        xtr4.set_ylabel('')    
        xtr4.set_xlabel('') 
        if ii == 0:
            xtr4.annotate('$probability_{event}$', xy=(3, 3), xytext=(12., -0.8),weight='bold',rotation=45,
                          fontsize=12)
            xtr4.annotate('$0.03$', xy=(0, 0), xytext=(12.5, 5),weight='bold',fontsize=12,bbox={"boxstyle" : "circle", "color":"whitesmoke"})
        if ii == 1:
            xtr4.annotate('$0.15$', xy=(3, 3), xytext=(12.5, 5),weight='bold',fontsize=12,bbox={"boxstyle" : "circle", "color":"whitesmoke"})
        if ii == 2:
            xtr4.annotate('$0.90$', xy=(3, 3), xytext=(12.5, 5),weight='bold',fontsize=12, bbox={"boxstyle" : "circle", "color":"whitesmoke"})

        
    fig.axes[1].set_title("$recovered_{proportion}$")
    fig.axes[3].set_title("$unintended_{count}$")
    fig.axes[5].set_title("$threshold_{FOM}}$")
    fig.axes[7].set_title("$SNR$")

    for i in range(0,23,2):
        fig.axes[i].set_xticks([])
        fig.axes[i].set_xticklabels([])
        fig.axes[i].set_yticks([])
        fig.axes[i].set_yticklabels([])


    fig.axes[16].set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5])
    fig.axes[16].set_xticklabels(scale)
    fig.axes[16].set_yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,])
    fig.axes[16].set_yticklabels(shape)
    fig.axes[16].set_ylabel('Shape')    
    fig.axes[16].set_xlabel('Scale') 

    fig.axes[9].set_visible(False)
    fig.axes[11].set_visible(False)
    fig.axes[13].set_visible(False)
    fig.axes[15].set_visible(False)
    fig.axes[17].set_visible(False)
    fig.axes[19].set_visible(False)
    fig.axes[21].set_visible(False)
    fig.axes[23].set_visible(False)
    plt.show()

#%%
def fig2(chars,loc,scale,shape):

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle
    import seaborn as sns

    fig = plt.figure(1, figsize=(12, 12))
    gs = gridspec.GridSpec(12, 24)
    gs.update(wspace=0.3, hspace=-.25)
    ys = [[0,2],[2,4],[4,6],[6,8],[8,10],[10,12]]
    which = 'loc'

    ts = chars[chars['which']=='s_sine']
    l = loc[0]#0.026
    tloc = ts[ts['loc']==l]
    cg = tloc.groupby(['p'])
    for ii,(which,g_) in enumerate(cg):
        
        th = g_.groupby(['shape','scale'])['ths'].mean().reset_index()
        th['ths'] = th['ths'] - 6    
        thp = th.pivot('shape', 'scale', 'ths')
        thpt = thp.T
        mids = np.argwhere((np.array(thpt)>-0.6) & (np.array(thpt)<0.6))
        
        f = g_.groupby(['shape','scale'])['quicks'].mean().reset_index()
        fp = f.pivot('shape', 'scale', 'quicks')
        xtr3 = fig.add_subplot(gs[ys[ii][0]:ys[ii][1], 6:12])
        xtr3 = sns.heatmap(fp, cmap="rocket_r", cbar=True,vmin=0,vmax=100,
                           cbar_kws = dict(use_gridspec=False,location="top",ticks=[0,100],shrink=0.6))
        for m in mids:
            xtr3.add_patch(Rectangle(m,0.9,0.9, fill=False, edgecolor='grey', lw=2, facecolor='grey', alpha=0.5,))
        xtr3.set_ylabel('')    
        xtr3.set_xlabel('') 
            
        xtr1 = fig.add_subplot(gs[ys[ii][0]:ys[ii][1], 12:18])
        xtr1 = sns.heatmap(thp, cmap="vlag",cbar = True, vmin=-6,vmax=6,
                    cbar_kws = dict(use_gridspec=False,location="top",ticks=[-6,6],shrink=0.6))
        c_bar = xtr1.collections[0].colorbar
        c_bar.set_ticks([-6, 0, 6])
        c_bar.set_ticklabels(['0', '6', '>12'])
        xtr1.set_ylabel('')    
        xtr1.set_xlabel('')     
        
        c = g_.groupby(['shape','scale'])['snr2'].mean().reset_index()
        cp = c.pivot('shape','scale', 'snr2')
        xtr4 = fig.add_subplot(gs[ys[ii][0]:ys[ii][1], 18:24])
        xtr4 = sns.heatmap(cp, cmap="rocket_r", cbar=True,vmin=1,vmax=5,
                           cbar_kws = dict(use_gridspec=False,location="top",shrink=0.6))
        c_bar = xtr4.collections[0].colorbar
        c_bar.set_ticks([1, 5])
        c_bar.set_ticklabels(['1', '5'])
        xtr4.set_ylabel('')    
        xtr4.set_xlabel('') 
        if ii == 0:
            xtr4.annotate('$probability_{event}$', xy=(3, 3), xytext=(12., -0.8),weight='bold',rotation=45,
                          fontsize=12)
            xtr4.annotate('$0.10$', xy=(0, 0), xytext=(12.5, 5),weight='bold',fontsize=12,bbox={"boxstyle" : "circle", "color":"whitesmoke"})
        if ii == 1:
            xtr4.annotate('$0.90$', xy=(3, 3), xytext=(12.5, 5),weight='bold',fontsize=12,bbox={"boxstyle" : "circle", "color":"whitesmoke"})
        
    fig.axes[1].set_title("$Events <300ms$")
    fig.axes[3].set_title("$threshold}$")
    fig.axes[5].set_title("$SNR}$")
    
    fig.axes[6].set_ylabel('Shape')    
    fig.axes[6].set_xlabel('Scale') 
    for i in range(0,11,2):
        fig.axes[i].set_xticks([])
        fig.axes[i].set_xticklabels([])
        fig.axes[i].set_yticks([])
        fig.axes[i].set_yticklabels([])


    fig.axes[6].set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5])
    fig.axes[6].set_xticklabels(scale)
    fig.axes[6].set_yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,])
    fig.axes[6].set_yticklabels(shape)

    fig.axes[7].set_visible(False)
    fig.axes[9].set_visible(False)
    fig.axes[11].set_visible(False)
    plt.show()   
#%%
def fig3(cut_hi_t,cut_mi_t,cut_lo_t,cut_hi_s,cut_lo_s):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import config as cfg
    import numpy as np
    chtp,chtpm,chtf,chtfm,chtd,chtdm = get_app_data(cut_hi_t)
    cmtp,cmtpm,cmtf,cmtfm,cmtd,cmtdm = get_app_data(cut_mi_t)
    cltp,cltpm,cltf,cltfm,cltd,cltdm = get_app_data(cut_lo_t)
    chsp,chspm,chsf,chsfm,chsd,chsdm = get_app_data(cut_hi_s)
    clsp,clspm,clsf,clsfm,clsd,clsdm = get_app_data(cut_lo_s)
    fig = plt.figure(1, figsize=(18, 18))
    gs = gridspec.GridSpec(26, 26)
    gs.update(wspace=7, hspace=5)
    sns.set_palette("flare")

    #threhsholds
    xtr_0_0 = fig.add_subplot(gs[0:3, 0:3])
    xtr_1_0 = fig.add_subplot(gs[3:6, 0:3])
    xtr_2_0 = fig.add_subplot(gs[6:9, 0:3])
    xtr_3_0 = fig.add_subplot(gs[9:12, 0:3])
    xtr_4_0 = fig.add_subplot(gs[12:15, 0:3])
    sns.set_palette("flare")
    fsize = 12
    for i,c in cut_lo_t.iterrows():
        sns.lineplot(x=cfg.thresholds,y=c['rsq'],palette='flare',ax=xtr_0_0)
        xtr_0_0.set_xticks([0, 6, 10, 15, 20])
        xtr_0_0.set_xticklabels(['0', '6','10','15', '20']) 
        xtr_0_0.set_yticks([0, 1])
        xtr_0_0.set_yticklabels(['0', '1'])
        xtr_0_0.set(xlabel=None,ylabel=None)
        xtr_0_0.set_title('Thresholds',fontsize=fsize)
        sns.despine(offset=10, trim=True, ax=xtr_0_0)
        
    for i,c in cut_mi_t.iterrows():
        sns.lineplot(x=cfg.thresholds,y=c['rsq'],palette='flare',ax=xtr_1_0)
        xtr_1_0.set_xticks([0, 6, 10, 15, 20])
        xtr_1_0.set_xticklabels(['0', '6','10','15', '20']) 
        xtr_1_0.set_yticks([0, 1])
        xtr_1_0.set_yticklabels(['0', '1'])
        xtr_1_0.set(xlabel=None,ylabel=None)
        sns.despine(offset=10, trim=True, ax=xtr_1_0)
        
    for i,c in cut_hi_t.iterrows():
        sns.lineplot(x=cfg.thresholds,y=c['rsq'],palette='flare',ax=xtr_2_0)
        xtr_2_0.set_xticks([0, 6, 10, 15, 20])
        xtr_2_0.set_xticklabels(['0', '6','10','15', '20']) 
        xtr_2_0.set_yticks([0, 1])
        xtr_2_0.set_yticklabels(['0', '1'])
        xtr_2_0.set(xlabel=None,ylabel=None)
        sns.despine(offset=10, trim=True, ax=xtr_2_0)

    for i,c in cut_lo_s.iterrows():
        sns.lineplot(x=cfg.thresholds,y=c['rsq'],palette='flare',ax=xtr_3_0)
        xtr_3_0.set_xticks([0, 6, 10, 15, 20])
        xtr_3_0.set_xticklabels(['0', '6','10','15', '20']) 
        xtr_3_0.set_yticks([0, 1])
        xtr_3_0.set_yticklabels(['0', '1'])
        xtr_3_0.set(xlabel=None,ylabel=None)
        sns.despine(offset=10, trim=True, ax=xtr_3_0)
        
    for i,c in cut_hi_s.iterrows():
        sns.lineplot(x=cfg.thresholds,y=c['rsq'],palette='flare',ax=xtr_4_0)
        xtr_4_0.set_xticks([0, 6, 10, 15, 20])
        xtr_4_0.set_xticklabels(['0', '6','10','15', '20']) 
        xtr_4_0.set_yticks([0, 1])
        xtr_4_0.set_yticklabels(['0', '1'])
        xtr_4_0.set(xlabel='Threshold',ylabel='Correlation')
        sns.despine(offset=10, trim=True, ax=xtr_4_0)
        
    #power
    xtr_0_1 = fig.add_subplot(gs[0:3, 3:6])
    xtr_1_1 = fig.add_subplot(gs[3:6, 3:6], sharey=xtr_0_1)
    xtr_2_1 = fig.add_subplot(gs[6:9, 3:6], sharey=xtr_0_1)
    xtr_3_1 = fig.add_subplot(gs[9:12, 3:6], sharey=xtr_0_1)
    xtr_4_1 = fig.add_subplot(gs[12:15, 3:6], sharey=xtr_0_1)

    yticks = [0,0.5]
    ylabel = ['0','0.5']
    sns.lineplot(data=cltp,x='bins_pow',y='pow', hue='index',ax=xtr_0_1,legend=False)
    sns.lineplot(data=cltpm,x='bins_pow',y='pow',linewidth = 3,ax=xtr_0_1)
    m = get_test(cut_lo_t)
    med = np.median([np.median(a) for a in m[0]])
    xtr_0_1.axvline(x=med,linewidth=2,color='grey',linestyle="--")
    xtr_0_1.set_xlim(0, 25)
    xtr_0_1.set_xticks([0,6,25])
    xtr_0_1.set_xticklabels(['0','6','25']) 
    xtr_0_1.set_ylim(0, 0.5) 
    xtr_0_1.set_yticks(yticks)
    xtr_0_1.set_yticklabels(ylabel)
    xtr_0_1.set(xlabel=None,ylabel=None)   
    xtr_0_1.set_title('Power',fontsize=fsize)
    sns.despine(offset=10, trim=True, ax=xtr_0_1)

    sns.lineplot(data=cmtp,x='bins_pow',y='pow', hue='index',ax=xtr_1_1,legend=False)
    sns.lineplot(data=cmtpm,x='bins_pow',y='pow',linewidth = 3,ax=xtr_1_1)
    m = get_test(cut_mi_t)
    med = np.median([np.median(a) for a in m[0]])
    xtr_1_1.axvline(x=med,linewidth=2, color='grey',linestyle="--")
    xtr_1_1.set_xlim(0, 25)
    xtr_1_1.set_xticks([0,6,25])
    xtr_1_1.set_xticklabels(['0','6','25'])  
    xtr_1_1.set_yticks(yticks)
    xtr_1_1.set_yticklabels(ylabel)
    xtr_1_1.set(xlabel=None,ylabel=None)    
    sns.despine(offset=10, trim=True, ax=xtr_1_1)

    sns.lineplot(data=chtp,x='bins_pow',y='pow', hue='index',ax=xtr_2_1,legend=False)
    sns.lineplot(data=chtpm,x='bins_pow',y='pow',linewidth = 3,ax=xtr_2_1)
    m = get_test(cut_hi_t)
    med = np.median([np.median(a) for a in m[0]])
    xtr_2_1.axvline(x=med,linewidth=2, color='grey',linestyle="--")
    xtr_2_1.set_xlim(0, 25)
    xtr_2_1.set_xticks([0,6,25])
    xtr_2_1.set_xticklabels(['0','6', '25'])  
    xtr_2_1.set_yticks(yticks)
    xtr_2_1.set_yticklabels(ylabel)
    xtr_2_1.set(xlabel=None,ylabel=None)    
    sns.despine(offset=10, trim=True, ax=xtr_2_1)

    sns.lineplot(data=clsp,x='bins_pow',y='pow', hue='index',ax=xtr_3_1,legend=False)
    sns.lineplot(data=clspm,x='bins_pow',y='pow',linewidth = 3,ax=xtr_3_1)
    m = get_test(cut_lo_s)
    med = np.median([np.median(a) for a in m[0]])
    xtr_3_1.axvline(x=med,linewidth=2, color='grey',linestyle="--")
    xtr_3_1.set_xlim(0, 25)
    xtr_3_1.set_xticks([0,6,25])
    xtr_3_1.set_xticklabels(['0','6','25'])  
    xtr_3_1.set_yticks(yticks)
    xtr_3_1.set_yticklabels(ylabel)
    xtr_3_1.set(xlabel=None,ylabel=None)    
    sns.despine(offset=10, trim=True, ax=xtr_3_1)

    sns.lineplot(data=chsp,x='bins_pow',y='pow', hue='index',ax=xtr_4_1,legend=False)
    sns.lineplot(data=chspm,x='bins_pow',y='pow',linewidth = 3,ax=xtr_4_1)
    m = get_test(cut_hi_s)
    med = np.median([np.median(a) for a in m[0]])
    xtr_4_1.axvline(x=med,linewidth=2, color='grey',linestyle="--")
    xtr_4_1.set_xlim(0, 25)
    xtr_4_1.set_xticks([0,6,25])
    xtr_4_1.set_xticklabels(['0','6','25'])  
    xtr_4_1.set_yticks(yticks)
    xtr_4_1.set_yticklabels(ylabel)
    xtr_4_1.set(xlabel=None,ylabel=None)    
    sns.despine(offset=10, trim=True, ax=xtr_4_1)

    #Frequency
    yticks2 = [0,0.8]
    ylabel2 = ['0','0.8']
    xticks = [0,20]
    xticklabels = ['0','20']

    xtr_0_2 = fig.add_subplot(gs[0:3, 6:9])
    xtr_1_2 = fig.add_subplot(gs[3:6, 6:9], sharey=xtr_0_2)
    xtr_2_2 = fig.add_subplot(gs[6:9, 6:9], sharey=xtr_0_2)
    xtr_3_2 = fig.add_subplot(gs[9:12, 6:9], sharey=xtr_0_2)
    xtr_4_2 = fig.add_subplot(gs[12:15, 6:9], sharey=xtr_0_2)

    sns.lineplot(data=cltf,x='bins_freq',y='freq', hue='index',ax=xtr_0_2,legend=False)
    sns.lineplot(data=cltfm,x='bins_freq',y='freq',linewidth = 3,ax=xtr_0_2)
    m = get_test(cut_lo_t)
    med = np.median([np.median(a) for a in m[1]])
    xtr_0_2.axvline(x=med,linewidth=2,color='grey',linestyle="--")
    xtr_0_2.set_xlim(0, 20)
    xtr_0_2.set_xticks(xticks)
    xtr_0_2.set_xticklabels(xticklabels)  
    xtr_0_2.set_yticks(yticks2)
    xtr_0_2.set_yticklabels(ylabel2)
    xtr_0_2.set(xlabel=None,ylabel=None)    
    xtr_0_2.set_title('Frequency Span',fontsize=fsize)
    sns.despine(offset=10, trim=True, ax=xtr_0_2)

    sns.lineplot(data=cmtf,x='bins_freq',y='freq', hue='index',ax=xtr_1_2,legend=False)
    sns.lineplot(data=cmtfm,x='bins_freq',y='freq',linewidth = 3,ax=xtr_1_2)
    m = get_test(cut_mi_t)
    med = np.median([np.median(a) for a in m[1]])
    xtr_1_2.axvline(x=med,linewidth=2,color='grey',linestyle="--")
    xtr_1_2.set_xlim(0, 20)
    xtr_1_2.set_xticks(xticks)
    xtr_1_2.set_xticklabels(xticklabels)  
    xtr_1_2.set_yticks(yticks2)
    xtr_1_2.set_yticklabels(ylabel2)
    xtr_1_2.set(xlabel=None,ylabel=None)    
    sns.despine(offset=10, trim=True, ax=xtr_1_2)

    sns.lineplot(data=chtf,x='bins_freq',y='freq', hue='index',ax=xtr_2_2,legend=False)
    sns.lineplot(data=chtfm,x='bins_freq',y='freq',linewidth = 3,ax=xtr_2_2)
    m = get_test(cut_hi_t)
    med = np.median([np.median(a) for a in m[1]])
    xtr_2_2.axvline(x=med,linewidth=2,color='grey',linestyle="--")
    xtr_2_2.set_xlim(0, 20)
    xtr_2_2.set_xticks(xticks)
    xtr_2_2.set_xticklabels(xticklabels)  
    xtr_2_2.set_yticks(yticks2)
    xtr_2_2.set_yticklabels(ylabel2)
    xtr_2_2.set(xlabel=None,ylabel=None)    
    sns.despine(offset=10, trim=True, ax=xtr_2_2)

    sns.lineplot(data=clsf,x='bins_freq',y='freq', hue='index',ax=xtr_3_2,legend=False)
    sns.lineplot(data=clsfm,x='bins_freq',y='freq',linewidth = 3,ax=xtr_3_2)
    m = get_test(cut_lo_s)
    med = np.median([np.median(a) for a in m[1]])
    xtr_3_2.axvline(x=med,linewidth=2,color='grey',linestyle="--")
    xtr_3_2.set_xlim(0, 20)
    xtr_3_2.set_xticks(xticks)
    xtr_3_2.set_xticklabels(xticklabels)  
    xtr_3_2.set_yticks(yticks2)
    xtr_3_2.set_yticklabels(ylabel2)
    xtr_3_2.set(xlabel=None,ylabel=None)    
    sns.despine(offset=10, trim=True, ax=xtr_3_2)

    sns.lineplot(data=chsf,x='bins_freq',y='freq', hue='index',ax=xtr_4_2,legend=False)
    sns.lineplot(data=chsfm,x='bins_freq',y='freq',linewidth = 3,ax=xtr_4_2)
    m = get_test(cut_hi_s)
    med = np.median([np.median(a) for a in m[1]])
    xtr_4_2.axvline(x=med,linewidth=2,color='grey',linestyle="--")
    xtr_4_2.set_xlim(0, 20)
    xtr_4_2.set_xticks(xticks)
    xtr_4_2.set_xticklabels(xticklabels)  
    xtr_4_2.set_yticks(yticks2)
    xtr_4_2.set_yticklabels(ylabel2)
    xtr_4_2.set(xlabel=None,ylabel=None)    
    sns.despine(offset=10, trim=True, ax=xtr_4_2)

    #Duration
    xtr_0_3 = fig.add_subplot(gs[0:3, 9:12])
    xtr_1_3 = fig.add_subplot(gs[3:6, 9:12], sharey=xtr_0_3)
    xtr_2_3 = fig.add_subplot(gs[6:9, 9:12], sharey=xtr_0_3)
    xtr_3_3 = fig.add_subplot(gs[9:12, 9:12], sharey=xtr_0_3)
    xtr_4_3 = fig.add_subplot(gs[12:15, 9:12], sharey=xtr_0_3)
    yticks3 = [0,0.8]
    ylabel3 = ['0','0.8']
    
    f = 20
    sns.lineplot(data=cltd,x='bins_dur',y='dur', hue='index',ax=xtr_0_3,legend=False)
    sns.lineplot(data=cltdm,x='bins_dur',y='dur',linewidth = 3,ax=xtr_0_3)
    m = get_test(cut_lo_t)
    med = np.median([np.median(a) for a in m[2]])*f
    xtr_0_3.axvline(x=med,linewidth=2,color='grey',linestyle="--")
    xtr_0_3.set_xlim(0, 12)
    xtr_0_3.set_xticks([0,12])
    xtr_0_3.set_xticklabels(['0', '0.6'])  
    xtr_0_3.set_yticks(yticks3)
    xtr_0_3.set_yticklabels(ylabel3)
    xtr_0_3.set(xlabel=None,ylabel=None)  
    xtr_0_3.set_title('Durations',fontsize=fsize)
    sns.despine(offset=10, trim=True, ax=xtr_0_3)

    sns.lineplot(data=cmtd,x='bins_dur',y='dur', hue='index',ax=xtr_1_3,legend=False)
    sns.lineplot(data=cmtdm,x='bins_dur',y='dur',linewidth = 3,ax=xtr_1_3)
    m = get_test(cut_mi_t)
    med = np.median([np.median(a) for a in m[2]])*f
    xtr_1_3.axvline(x=med,linewidth=2,color='grey',linestyle="--")
    xtr_1_3.set_xlim(0, 12)
    xtr_1_3.set_xticks([0,12])
    xtr_1_3.set_xticklabels(['0', '0.6'])  
    xtr_1_3.set_yticks(yticks3)
    xtr_1_3.set_yticklabels(ylabel3)
    xtr_1_3.set(xlabel=None,ylabel=None) 
    sns.despine(offset=10, trim=True, ax=xtr_1_3)

    sns.lineplot(data=chtd,x='bins_dur',y='dur', hue='index',ax=xtr_2_3,legend=False)
    sns.lineplot(data=chtdm,x='bins_dur',y='dur',linewidth = 3,ax=xtr_2_3)
    m = get_test(cut_hi_t)
    med = np.median([np.median(a) for a in m[2]])*f
    xtr_2_3.axvline(x=med,linewidth=2,color='grey',linestyle="--")
    xtr_2_3.set_xlim(0, 12)
    xtr_2_3.set_xticks([0,12])
    xtr_2_3.set_xticklabels(['0', '0.6'])  
    xtr_2_3.set_yticks(yticks3 )
    xtr_2_3.set_yticklabels(ylabel3)
    xtr_2_3.set(xlabel=None,ylabel=None)    
    sns.despine(offset=10, trim=True, ax=xtr_2_3)

    sns.lineplot(data=clsd,x='bins_dur',y='dur', hue='index',ax=xtr_3_3,legend=False)
    sns.lineplot(data=clsdm,x='bins_dur',y='dur',linewidth = 3,ax=xtr_3_3)
    m = get_test(cut_lo_s)
    med = np.median([np.median(a) for a in m[2]])*f
    xtr_3_3.axvline(x=med,linewidth=2,color='grey',linestyle="--")
    xtr_3_3.set_xlim(0, 12)
    xtr_3_3.set_xticks([0,12])
    xtr_3_3.set_xticklabels(['0', '0.6'])  
    xtr_3_3.set_yticks(yticks3 )
    xtr_3_3.set_yticklabels(ylabel3)
    xtr_3_3.set(xlabel=None,ylabel=None)    
    sns.despine(offset=10, trim=True, ax=xtr_3_3)

    sns.lineplot(data=chsd,x='bins_dur',y='dur', hue='index',ax=xtr_4_3,legend=False)
    sns.lineplot(data=chsdm,x='bins_dur',y='dur',linewidth = 3,ax=xtr_4_3)
    m = get_test(cut_hi_s)
    med = np.median([np.median(a) for a in m[2]])*f
    xtr_4_3.axvline(x=med,linewidth=2,color='grey',linestyle="--")
    xtr_4_3.set_xlim(0, 12)
    xtr_4_3.set_xticks([0,12])
    xtr_4_3.set_xticklabels(['0', '0.6'])  
    xtr_4_3.set_yticks(yticks3 )
    xtr_4_3.set_yticklabels(ylabel3)
    xtr_4_3.set(xlabel=None,ylabel=None)    
    sns.despine(offset=10, trim=True, ax=xtr_4_3)

    xtr_0_4 = fig.add_subplot(gs[0:3, 12:15])
    xtr_1_4 = fig.add_subplot(gs[3:6, 12:15], sharey=xtr_0_4)
    xtr_2_4 = fig.add_subplot(gs[6:9, 12:15], sharey=xtr_0_4)
    xtr_3_4 = fig.add_subplot(gs[9:12, 12:15], sharey=xtr_0_4)
    xtr_4_4 = fig.add_subplot(gs[12:15, 12:15], sharey=xtr_0_4)

    xticks = [-0.5, 5.5]
    xticklabels = ['0', '5']
    yticks = [0, 100]
    yticklabels = ['0', '100']
    # counts are 4 and 11
    for i,c in cut_lo_t.iterrows():
        xtr_0_4.hist(c[4],bins=np.linspace(-1, 10, 10), 
            histtype=u'bar', density=False)
        xtr_0_4.set_xticks(xticks)
        xtr_0_4.set_xticklabels(xticklabels)   
        xtr_0_4.set_yticks(yticks)
        xtr_0_4.set_yticklabels(yticklabels) 
        xtr_0_4.set_title('Detected Count',fontsize=fsize)
        sns.despine(offset=10, trim=True, ax=xtr_0_4)
        
    for i,c in cut_mi_t.iterrows():
        xtr_1_4.hist(c[4],bins=np.linspace(-1, 10, 10), 
            histtype=u'bar', density=False)
        xtr_1_4.set_xticks(xticks)
        xtr_1_4.set_xticklabels(xticklabels)   
        xtr_1_4.set_yticks(yticks)
        xtr_1_4.set_yticklabels(yticklabels) 
        sns.despine(offset=10, trim=True, ax=xtr_1_4)
        
    for i,c in cut_hi_t.iterrows():
        xtr_2_4.hist(c[4],bins=np.linspace(-1, 10, 10), 
            histtype=u'bar', density=False)
        xtr_2_4.set_xticks(xticks)
        xtr_2_4.set_xticklabels(xticklabels)   
        xtr_2_4.set_yticks(yticks)
        xtr_2_4.set_yticklabels(yticklabels) 
        sns.despine(offset=10, trim=True, ax=xtr_2_4)
        
    for i,c in cut_lo_s.iterrows():
        xtr_3_4.hist(c[4],bins=np.linspace(-1, 10, 10), 
            histtype=u'bar', density=False)
        xtr_3_4.set_xticks(xticks)
        xtr_3_4.set_xticklabels(xticklabels)   
        xtr_3_4.set_yticks(yticks)
        xtr_3_4.set_yticklabels(yticklabels) 
        sns.despine(offset=10, trim=True, ax=xtr_3_4)
        
    for i,c in cut_hi_s.iterrows():
        xtr_4_4.hist(c[4],bins=np.linspace(-1, 10, 10), 
            histtype=u'bar', density=False)
        xtr_4_4.set_xticks(xticks)
        xtr_4_4.set_xticklabels(xticklabels)   
        xtr_4_4.set_yticks(yticks)
        xtr_4_4.set_yticklabels(yticklabels) 
        sns.despine(offset=10, trim=True, ax=xtr_4_4)
        
    xtr_0_a = fig.add_subplot(gs[0:3, 15:18])
    xtr_1_a = fig.add_subplot(gs[3:6, 15:18], sharey=xtr_0_4)
    xtr_2_a = fig.add_subplot(gs[6:9, 15:18], sharey=xtr_0_4)
    xtr_3_a = fig.add_subplot(gs[9:12, 15:18], sharey=xtr_0_4)
    xtr_4_a = fig.add_subplot(gs[12:15, 15:18], sharey=xtr_0_4)

    xticks = [-0.5, 5.5]
    xticklabels = ['0', '5']
    yticks = [0, 100]
    yticklabels = ['0', '100']
    # counts are 4 and 11
    for i,c in cut_lo_t.iterrows():
        xtr_0_a.hist(c[11],bins=np.linspace(-1, 10, 10), 
            histtype=u'bar', density=False)
        xtr_0_a.set_xticks(xticks)
        xtr_0_a.set_xticklabels(xticklabels)   
        xtr_0_a.set_yticks(yticks)
        xtr_0_a.set_yticklabels(yticklabels) 
        xtr_0_a.set_title('Intended Count',fontsize=fsize)
        sns.despine(offset=10, trim=True, ax=xtr_0_a)
        
    for i,c in cut_mi_t.iterrows():
        xtr_1_a.hist(c[11],bins=np.linspace(-1, 10, 10), 
            histtype=u'bar', density=False)
        xtr_1_a.set_xticks(xticks)
        xtr_1_a.set_xticklabels(xticklabels)   
        xtr_1_a.set_yticks(yticks)
        xtr_1_a.set_yticklabels(yticklabels) 
        sns.despine(offset=10, trim=True, ax=xtr_1_a)
        
    for i,c in cut_hi_t.iterrows():
        xtr_2_a.hist(c[11],bins=np.linspace(-1, 10, 10), 
            histtype=u'bar', density=False)
        xtr_2_a.set_xticks(xticks)
        xtr_2_a.set_xticklabels(xticklabels)   
        xtr_2_a.set_yticks(yticks)
        xtr_2_a.set_yticklabels(yticklabels) 
        sns.despine(offset=10, trim=True, ax=xtr_2_a)
        
    for i,c in cut_lo_s.iterrows():
        xtr_3_a.hist(c[11],bins=np.linspace(-1, 10, 10), 
            histtype=u'bar', density=False)
        xtr_3_a.set_xticks(xticks)
        xtr_3_a.set_xticklabels(xticklabels)   
        xtr_3_a.set_yticks(yticks)
        xtr_3_a.set_yticklabels(yticklabels) 
        sns.despine(offset=10, trim=True, ax=xtr_3_a)
        
    for i,c in cut_hi_s.iterrows():
        xtr_4_a.hist(c[11],bins=np.linspace(-1, 10, 10), 
            histtype=u'bar', density=False)
        xtr_4_a.set_xticks(xticks)
        xtr_4_a.set_xticklabels(xticklabels)   
        xtr_4_a.set_yticks(yticks)
        xtr_4_a.set_yticklabels(yticklabels) 
        sns.despine(offset=10, trim=True, ax=xtr_4_a)    
    #avg    
    xtr_0_5 = fig.add_subplot(gs[0:3, 18:22])
    xtr_1_5 = fig.add_subplot(gs[3:6, 18:22])
    xtr_2_5 = fig.add_subplot(gs[6:9, 18:22])
    xtr_3_5 = fig.add_subplot(gs[9:12, 18:22])
    xtr_4_5 = fig.add_subplot(gs[12:15, 18:22])
    xticks = [0, 50,100]
    xticklabels = ['-0.2','0','0.2']
    yticks = [-0.1, 0.1]
    yticklabels = ['-0.1', '0.1']

    for i,c in cut_lo_t.iterrows():    
        xtr_0_5.plot(c[7][25:125])
        xtr_0_5.set_xticks(xticks)
        xtr_0_5.set_xticklabels(xticklabels)   
        xtr_0_5.set_yticks(yticks)
        xtr_0_5.set_yticklabels(yticklabels) 
        xtr_0_5.set_title('BURPS',fontsize=fsize)
        sns.despine(offset=10, trim=True, ax=xtr_0_5)

    for i,c in cut_mi_t.iterrows():    
        xtr_1_5.plot(c[7][25:125])
        xtr_1_5.set_xticks(xticks)
        xtr_1_5.set_xticklabels(xticklabels)   
        xtr_1_5.set_yticks(yticks)
        xtr_1_5.set_yticklabels(yticklabels) 
        sns.despine(offset=10, trim=True, ax=xtr_1_5)
        
    for i,c in cut_hi_t.iterrows():    
        xtr_2_5.plot(c[7][25:125])
        xtr_2_5.set_xticks(xticks)
        xtr_2_5.set_xticklabels(xticklabels)   
        xtr_2_5.set_yticks(yticks)
        xtr_2_5.set_yticklabels(yticklabels) 
        sns.despine(offset=10, trim=True, ax=xtr_2_5)
    m =[]    
    for i,c in cut_lo_s.iterrows(): 
        m.append(c[7][25:125])
        xtr_3_5.plot(c[7][25:125])
        xtr_3_5.set_xticks(xticks)
        xtr_3_5.set_xticklabels(xticklabels)   
        xtr_3_5.set_yticks(yticks)
        xtr_3_5.set_yticklabels(yticklabels) 
        sns.despine(offset=10, trim=True, ax=xtr_3_5)
     
    m = []
    for i,c in cut_hi_s.iterrows():   
        m.append(c[7][25:125])
        xtr_4_5.plot(c[7][25:125])
        xtr_4_5.set_xticks(xticks)
        xtr_4_5.set_xticklabels(xticklabels)   
        xtr_4_5.set_yticks(yticks)
        xtr_4_5.set_yticklabels(yticklabels) 
        sns.despine(offset=10, trim=True, ax=xtr_4_5)
    
    xtr_0_5.annotate('$probability_{event}$', xy=(0, 0), xytext=(100, 0.1),weight='bold',rotation=45,
                          fontsize=15)
    xtr_0_5.annotate('$0.03$', xy=(0, 0), xytext=(120, 0),weight='bold',fontsize=15,bbox={"boxstyle" : "circle", "color":"whitesmoke"})
    xtr_1_5.annotate('$0.15$', xy=(0, 0), xytext=(120, 0),weight='bold',fontsize=15,bbox={"boxstyle" : "circle", "color":"whitesmoke"})
    xtr_2_5.annotate('$0.90$', xy=(0, 0), xytext=(120, 0),weight='bold',fontsize=15,bbox={"boxstyle" : "circle", "color":"whitesmoke"})
    xtr_3_5.annotate('$0.10$', xy=(0, 0), xytext=(120, 0),weight='bold',fontsize=15,bbox={"boxstyle" : "circle", "color":"whitesmoke"})
    xtr_4_5.annotate('$0.90$', xy=(0, 0), xytext=(120, 0),weight='bold',fontsize=15,bbox={"boxstyle" : "circle", "color":"whitesmoke"})

    draw_brace(xtr_1_0, 3, (-10,0.5), 'Transient', (-18,0.1), brace_scale=5.0, beta_scale=9500., rotate=True, rotate_text=True)
    draw_brace(xtr_4_0, 1.65, (-10,1.3), 'Sustained', (-18,1), brace_scale=5.0, beta_scale=9500., rotate=True, rotate_text=True)

    xtr_4_0.set(xlabel='Threshold',ylabel='Correlation')
    xtr_4_0.set_ylabel('Correlation', labelpad=-10)
    xtr_4_1.set(xlabel='Power',ylabel='Probability')
    xtr_4_1.set_ylabel('Probability', labelpad=-20)
    xtr_4_2.set(xlabel='Frequency Span (Hz)',ylabel='Probability')
    xtr_4_2.set_ylabel('Probability', labelpad=-20)
    xtr_4_3.set(xlabel='Duration (s)',ylabel='Probability')
    xtr_4_3.set_ylabel('Probability', labelpad=-20)
    xtr_4_4.set(xlabel='Detected',ylabel='Trial Count')
    xtr_4_4.set_ylabel('Trial Count', labelpad=-20)
    xtr_4_a.set(xlabel='Intended',ylabel='Trial Count')
    xtr_4_a.set_ylabel('Trial Count', labelpad=-20)
    xtr_4_5.set(xlabel='Time (s)',ylabel='Amplitude')
    xtr_4_5.set_ylabel('Amplitude', labelpad=-2,rotation='horizontal')  
    xtr_4_5.yaxis.set_label_coords(-0,0.7)

    c = 'whitesmoke'
    xtr_3_0.set_facecolor(c)
    xtr_4_0.set_facecolor(c)
    xtr_3_1.set_facecolor(c)
    xtr_4_1.set_facecolor(c)    
    xtr_3_2.set_facecolor(c)
    xtr_4_2.set_facecolor(c)
    xtr_3_3.set_facecolor(c)
    xtr_4_3.set_facecolor(c)
    xtr_3_4.set_facecolor(c)
    xtr_4_4.set_facecolor(c)
    xtr_3_a.set_facecolor(c)
    xtr_4_a.set_facecolor(c)    
    xtr_3_5.set_facecolor(c)
    xtr_4_5.set_facecolor(c)
    
    plt.show()
