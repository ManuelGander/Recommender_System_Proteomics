import pandas as pd
import numpy as np
import seaborn as sns
import anndata
import matplotlib.pyplot as plt
import scipy


import os
import sys
sys.path.append(os.getcwd().rsplit('/', 1)[0]+'/Functions')

import als
import neural_network_RS as nn



def partition_matrix(matrix):
    n0,n1=matrix.shape
    
    S={'Unknown':[], 'Empty':[], 'Not regulated':[], 'Up-regulated':[], 'Down-regulated':[]}

    for i in range(n0):
        for j in range(n1):
            if matrix.X[i,j] in S.keys():
                S[matrix.X[i,j]].append((i,j))
            else:
                print(f"{(i,j)} leads to an undefined regualtion state: {matrix.X[i,j]}")
    return(S)


def define_train_and_test_set(S, ratio=0.05):
    D={}
    D['Train']={}
    D['Test']={}

    l=0
    for key in S.keys():
        np.random.seed(l)
        l+=1

        v=S[key]
        np.random.shuffle(v)

        D['Train'][key]=v[:round(len(v)*(1-ratio))]
        D['Test'][key]=v[round(len(v)*(1-ratio)):]
    return(D)



def variant_to_set(Ds, variant):
    
    neg_set=Ds['Not regulated']

    if variant=='up':
        pos_set=Ds['Up-regulated']
        neg_set=[*neg_set, *Ds['Down-regulated']]
    elif variant=='down':
        pos_set=Ds['Down-regulated']
        neg_set=[*neg_set, *Ds['Up-regulated']]
    elif variant=='up+down':
        pos_set=[*Ds['Down-regulated'], *Ds['Up-regulated']]
    else:
        print('Variant not defined (must be in "up", "down", "up+down")')
    return(neg_set, pos_set)



def evaluate_test_set(D, C, variant, plot_hists=True, nn=False):
    not_reg=[]
    reg=[]

    for i in D['Test']['Not regulated']:
        not_reg.append(C[i])

    if variant=='up':
        for i in D['Test']['Up-regulated']:
            reg.append(C[i])
        for i in D['Test']['Down-regulated']:
            not_reg.append(C[i])

    elif variant=='down':
        for i in D['Test']['Down-regulated']:
            reg.append(C[i])
        for i in D['Test']['Up-regulated']:
            not_reg.append(C[i])

    elif variant=='up+down':
        for i in D['Test']['Down-regulated']:
            reg.append(C[i])
        for i in D['Test']['Up-regulated']:
            reg.append(C[i])
    
    df_not_reg=pd.DataFrame({'Score':not_reg, 'Set':'Not regulated'})
    df_reg=pd.DataFrame({'Score':reg, 'Set':f'Regulated ({variant})'})
    df=pd.concat([df_not_reg, df_reg])
    df.index=np.arange(len(df))
    
    if plot_hists:
        
        l=[]
        for i in D['Test']['Unknown']:
            l.append(C[i])
        df_unknowns=pd.DataFrame({'Distribution of scores for the different sets':l, 'Set':'Unknown'})


        sns.set(rc={'figure.figsize':(12,5)})
        sns.set_style('darkgrid')

        plt.subplot(2, 1, 1)

        sns.set(rc={'figure.figsize':(12,5)})
        
        
        if nn:
            bins=np.linspace(0.2,1.2,101)
        else:
            bins=np.linspace(-1.7,1.2,101)
        
        sns.histplot(data=df, x='Score', hue='Set', log_scale=(False, False),bins=bins)
        plt.ylim(0, 300)

        plt.subplot(2, 1, 2)

        sns.histplot(data=df_unknowns, x='Distribution of scores for the different sets', hue='Set', bins=bins)

    return(df)



def errors_given_cutoff(df, cut):
    df_not_reg=df[df['Set']=='Not regulated']

    variant=df[df['Set']!='Not regulated']['Set'].iloc[0].split('(')[1][:-1]

    df_reg=df[df['Set']==f'Regulated ({variant})']


    if variant!='down':

        false_positives=np.array(df_not_reg['Score']>cut).sum()
        true_negatives=np.array(df_not_reg['Score']<=cut).sum()

        false_negatives=np.array(df_reg['Score']<cut).sum()
        true_positives=np.array(df_reg['Score']>=cut).sum()

    else:
        false_positives=np.array(df_not_reg['Score']<cut).sum()
        true_negatives=np.array(df_not_reg['Score']>=cut).sum()

        false_negatives=np.array(df_reg['Score']>cut).sum()
        true_positives=np.array(df_reg['Score']<=cut).sum()

    precission=true_positives/(true_positives+false_positives)
    recall=true_positives/(true_positives+false_negatives)

    return(precission, recall)

def precission_recall_curve(df, bins=201, plot=True, zoom_in=False):
    c=[]
    p=[]
    r=[]

    for cut in np.linspace(-1.5,1.5,bins):

        precission, recall = errors_given_cutoff(df, cut)

        if not np.isnan(precission) and not np.isnan(recall):
            c.append(cut)
            p.append(precission)
            r.append(recall)

    df_precission_recall=pd.DataFrame({'Cutoff':c, 'Precission':p, 'Recall':r, 'Method':'ALS'})
    
    if plot:
    
        sns.set_style("darkgrid")
        sns.set(rc={'figure.figsize':(9,6)})

        fig = plt.figure()

        sns.lineplot(data=df_precission_recall, x='Recall', y='Precission', hue='Method')

        if zoom_in:
            plt.ylim(0.7, 1.0)
            plt.xlim(0.7, 1.0)
        else:
            plt.ylim(-0.05, 1.05)
            plt.xlim(-0.05, 1.05)
        
    return(df_precission_recall)



def plot_precision_recall_comparison(matrix):
    df0=matrix.uns['ALS']
    df0['Method']='ALS'
    df1=matrix.uns['NN_Additional_features']
    df1['Method']='NN_Additional_features'
    dfc=pd.concat([df0, df1])
    dfc.index=np.arange(len(dfc))
    
    sns.lineplot(data=dfc, x='Recall', y='Precission', hue='Method')
    plt.show()
    return(dfc)


def run_weighted_ALS(matrix, variant='up+down', test_set_ratio=0.05, dims=20, regularization=10, 
                     iterations=50, weight_empty=0.01, weight_unknown=0.01):
    S=partition_matrix(matrix)
    D=define_train_and_test_set(S, ratio=test_set_ratio)
    X=construct_train_matrix(matrix, D, variant=variant)
    C,X0,Y0=als.run_ALS_with_weights(matrix, D, X, dims, regularization, iterations, 
                                 weight_empty, weight_unknown)
    df=evaluate_test_set(D,C,variant)
    df_pr=precission_recall_curve(df, bins=10**3, plot=False, zoom_in=False)
    plt.pause(0.1)
    matrix.layers['ALS']=C
    matrix.uns['ALS']=df_pr
    matrix.varm['Drug_embedding']=Y0.T
    matrix.obsm['Peptide_embedding']=X0.T
    return(matrix)


def run_NN_RS(matrix, variant='up+down', test_set_ratio=0.05, epochs=3*10**2, dim0=20, dim1=5, dim_dl=3, dim_features=2, dim_dl_and_features=3):
    from tensorflow.keras import models, layers, utils  #(2.6.0)
    S=partition_matrix(matrix)
    D=define_train_and_test_set(S, ratio=test_set_ratio)   
    in0, in1, out=restructure_train_set(D, variant)
    EC50, SN, FC=nn.get_additional_information(matrix, in0, in1)
    model=nn.mf_and_dl_and_additional_features(matrix, dim0, dim1, dim_dl, dim_features, dim_dl_and_features)
    utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    training = model.fit(x=[in0, in1, EC50, SN, FC], y=out, epochs=3*10**2, batch_size=10**6, shuffle=True, verbose=0, validation_split=0.1)   
    C=nn.model_to_output_test_set(matrix, D, model, test_only=False) 
    df=evaluate_test_set(D,C,variant, nn=True)
    df_pr=precission_recall_curve(df, bins=10**3, plot=False, zoom_in=False)
    plt.pause(0.1)
    matrix.layers['NN_Additional_features']=C
    matrix.uns['NN_Additional_features']=df_pr
    return(matrix, model)




def construct_train_matrix(matrix, D, variant, neg=-1, pos=1):
    n0,n1=matrix.shape
    X=np.zeros((n0,n1))
    
    Ds=D['Train']
    neg_set, pos_set=variant_to_set(Ds, variant)       

    M=matrix.X
    for i in neg_set:
        X[i]=neg
    for i in pos_set:
        X[i]=pos
    return(X)

def construct_train_matrix_correlation(matrix, D, variant, key, initialization):
    n0,n1=matrix.shape
    
    Ds=D['Train']
    neg_set, pos_set=variant_to_set(Ds, variant)     
    
    X=np.full((n0,n1), initialization, dtype=float)
    
    if key in matrix.layers.keys():
        if key!='pEC50' and key!='log(Fold_change_predicted)':
            for i in [*neg_set, *pos_set]:
                X[i]=np.log(float(matrix.layers[key][i]))
        else:
            for i in [*neg_set, *pos_set]:
                X[i]=float(matrix.layers[key][i])
    else:
        print(f'Error: key {key} not in matrix.layers.keys()')

    return(X)



def restructure_train_set(D, variant, neg=0.5, pos=1):
    neg_set, pos_set=variant_to_set(D['Train'], variant)

    in0=[]
    in1=[]
    out=[]


    Dt=D['Train']

    for i in neg_set:
        in0.append(i[0])
        in1.append(i[1])
        out.append(neg)

    for i in pos_set:
        in0.append(i[0])
        in1.append(i[1])
        out.append(pos)

    in0=np.array(in0)
    in1=np.array(in1)
    out=np.array(out)
    p = np.random.permutation(len(in0))
    
    in0=in0[p]
    in1=in1[p]
    out=out[p]
    
    return(in0, in1, out)