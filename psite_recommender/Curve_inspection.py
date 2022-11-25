import pandas as pd
import numpy as np
import seaborn as sns
import anndata
import matplotlib.pyplot as plt
import scipy

import os
import sys
sys.path.append(os.getcwd().rsplit('/', 1)[0]+'/Functions')

import Extract_from_raw_data as extr


doses = [0.0, 0.03, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 10000.0]


def cutoff_given_precission(matrix, key, min_precission):
    df=matrix.uns[key]
    precission=np.array([a for a in df['Precission']])

    pr2=precission-min_precission
    pr2=[1 if a<0 else a for a in pr2]
    ind=np.argmin(pr2)

    cutoff, prec, recall=[a for a in df.iloc[ind]][:3]
    return(cutoff, prec, recall)

def recommend_peptide_drug_pairs(matrix, key, min_precision):
    cutoff, prec, recall=cutoff_given_precission(matrix, key, min_precision)
    
    n0,n1=matrix.shape
    
    recommended=[]
    for i in range(n0):
        for j in range(n1):
            if matrix.layers[key][i,j]>cutoff:
                recommended.append((i,j))
    return(recommended)


def select_only_identified(matrix, recommended):
    identified=[]

    for i in recommended:
        if matrix.X[i]=='Down-regulated' or matrix.X[i]=='Up-regulated' or matrix.X[i]=='Not regulated':

            dr=matrix.var.iloc[i[1]]['Drug']
            pept=matrix.obs.iloc[i[0]]['Modified sequence']
            df=Data[dr]

            if pept in df['Modified sequence']:
                identified.append(i)
    return(identified)


def get_curves(matrix, Data, indices, pick_criterion='Andromeda Score'):
    
    frames=[]
    
    if 'Empty' in matrix.X:
        for i in indices:
            if matrix.X[i]!='Empty':
                
                dr=matrix.var.iloc[i[1]]['Drug']
                pept=matrix.obs.iloc[i[0]]['Modified sequence']
                df=Data[dr]

                dfs=df[df['Modified sequence']==pept]
                dfs=extr.pick_curve(dfs, pick_criterion=pick_criterion).copy()
                frames.append(dfs)
        
    else:
        for i in indices:
            dr=matrix.var.iloc[i[1]]['Drug']
            pept=matrix.obs.iloc[i[0]]['Modified sequence']
            df=Data[dr]

            if pept in [a for a in df['Modified sequence']]:
                dfs=df[df['Modified sequence']==pept]
                dfs=extr.pick_curve(dfs, pick_criterion=pick_criterion).copy()
                frames.append(dfs)

    Df=pd.concat(frames)
    Df.index=np.arange(len(Df))
    Df['log2(new_FC)']=np.log2(Df['Fold_change_predicted'])
    return(Df)



def regulation_for_recommended_curves(df, alpha=0.03, fc_lim=0.35, new_not_regualted=True):

    Fold_change_key='Fold_change_predicted'
    Q_value_key='Curve Log P_Value'

    ############## This is a reduced version of Flo's "define_regulated_curves"    ##################################

    fold_change = np.log2(df[Fold_change_key].copy())
    s0 = extr.get_s0(fc_lim, alpha)
    p_value_cutoff = extr.map_fc_to_p_samcutoff(fold_change, alpha=alpha, s0=s0, deg_f=10-2)
    p_mask = df[Q_value_key] >= p_value_cutoff
    down_mask = fold_change < 0
    up_mask = fold_change > 0

    # Not regulated curves require: not too much noise, not too much deviation to control
    # TODO: Do we need a p value cut off ? currently up and down regulation will overwrite not,
    # also p value might be strong but no effect
    rmse_max = 0.1
    
    if new_not_regualted:
        not_regulated_mask = (df['Curve Simple Model RMSE'] < rmse_max) & \
                             (np.log2(df[Fold_change_key])).between(-fc_lim, fc_lim)
    
    else:
        not_regulated_mask = (df['Curve Simple Model RMSE'] < rmse_max) & \
                             (np.log2(df['Curve Simple Model'])).between(-abs(fc_lim)/2, abs(fc_lim)/2)


    # Add labels
    df['Regulation_after_RS'] = 'Unknown'
    df.loc[not_regulated_mask, 'Regulation_after_RS'] = 'Not regulated'
    df.loc[p_mask & up_mask, 'Regulation_after_RS'] = 'Up-regulated'
    df.loc[p_mask & down_mask, 'Regulation_after_RS'] = 'Down-regulated'
    
    return(df)


def vulcano_plot_recommender_curves(DF, alpha=0.03, fc_lim=0.35, plot_type='scatter', color='New_curve_regulation'):
    sns.set_style('darkgrid')
    sns.set(rc={'figure.figsize':(12,5)})

    deg_f=10-2
    x=np.linspace(fc_lim,7,100)
    y=extr.sam_curve(x, deg_f, fc_lim, alpha)

    x=[*-x,*x]
    y=[*y,*y]
    
    if plot_type=='scatter':
        # coloring with Drug and Confidence seems to only work with scatter plot and not with kde
        
        sns.scatterplot(data=DF, y='Curve Log P_Value', x='log2(new_FC)', s=10, hue='New_curve_regulation')
        
    elif plot_type=='kde':
        sns.kdeplot(data=DF, y='Curve Log P_Value', x='log2(new_FC)', s=10, hue='New_curve_regulation', cmap="Blues", shade=True, bw_adjust=0.2)
    
    else:
        print('Error: plot_type is not in [scatter, kde]')
    
    sns.lineplot(x=x,y=y, color='brown')
    
    
def apply_thresholds(Df, fc_lim=0.55, alpha=0.015, plot=True):
    log_alpha=-np.log10(alpha)

    sns.set_style('darkgrid')
    sns.set(rc={'figure.figsize':(12,5)})

    color='New_curve_regulation'
    Df_rec=Df[abs(Df['log2(new_FC)'])>fc_lim].copy()
    Df_rec=Df_rec[Df_rec['Curve Log P_Value']>log_alpha].copy()
    
    Df['-log10(p-value curve fit)']=Df['Curve Log P_Value']
    Df['log2(Fold change)']=Df['log2(new_FC)']
    Df['Curve regulation']=Df['New_curve_regulation']
    
    
    
    if plot:
        print(f'{round(len(Df_rec)/len(Df)*1000)/10}% of recommended curves ({len(Df_rec)} curves) satisfy the more leniant leniant classification criteria')
        deg_f=10-2
        x=np.linspace(fc_lim,7,1000)
        y=extr.sam_curve(x, deg_f, fc_lim, alpha)

        x=[*-x,*x]
        y=[*y,*y]

        sns.kdeplot(data=Df, y='-log10(p-value curve fit)', x='log2(Fold change)', s=10, hue='Curve regulation', cmap="Blues", shade=True, bw_adjust=1)

        sns.lineplot(x=x,y=y, color='brown')

        sns.lineplot(x=[-fc_lim,-fc_lim], y=[log_alpha, 10], color='brown', estimator=None, linewidth = 2)
        sns.lineplot(x=[fc_lim,fc_lim], y=[log_alpha, 10], color='brown', estimator=None, linewidth = 2)
        sns.lineplot(x=[-3,-fc_lim], y=[log_alpha, log_alpha], color='brown')
        sns.lineplot(x=[fc_lim,3], y=[log_alpha, log_alpha], color='brown')

        plt.ylim(0,8)
        plt.xlim(-3,3)
    
    return(Df_rec)       


def recommend_for_segmented(matrix, key='up', min_precission=0.95):
    c_l,a,b=cutoff_given_precission(matrix, 'dfdo', min_precission)
    c_u,a,b=cutoff_given_precission(matrix, 'dfup', min_precission)

    n0,n1=matrix.shape

    recommend=[]
    for i in range(n0):
        for j in range(n1):
            if matrix.layers['ALS_weighted'][i,j]>c_u:
                recommend.append((i,j))
                
    if key=='down':
        print('Recommending for down-regulation')


        recommend=[]
        for i in range(n0):
            for j in range(n1):
                if matrix.layers['ALS_weighted'][i,j]<c_l:
                    recommend.append((i,j))
    else:
        print('Recommending for up-regulation')
    return(recommend)



def recommend_curves(matrix, Data, rs_used='ALS', min_precision=0.95, unknowns_only=True):
    recommended=recommend_peptide_drug_pairs(matrix, rs_used, min_precision)
    
    if unknowns_only:
        indices=select_only_unknowns(matrix, recommended)
    else:
        indices=recommended
    
    Df=get_curves(matrix, Data, indices, pick_criterion='Andromeda Score')
    
    Dfu=Df[Df['New_curve_regulation']=='Unknown']
    
    Dfr=apply_thresholds(Dfu, plot=True)
    return(Dfr)


def select_only_unknowns(matrix, recommended):
    unknowns=[]
    for i in recommended:
        if matrix.X[i]=='Unknown':
            unknowns.append(i)
    return(unknowns)
