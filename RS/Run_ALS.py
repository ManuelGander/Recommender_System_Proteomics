import pandas as pd
import numpy as np
import seaborn as sns
import anndata
from scipy import sparse
import matplotlib.pyplot as plt

import implicit




def partition_matrix(matrix):
    l0=len(matrix.obs)
    l1=len(matrix.var)
    
    D={}
    D['Unknown']=[]
    D['Not_regulated']=[]
    D['Up']=[]
    D['Down']=[]

    for i in range(l0):
        for j in range(l1):
            if np.isnan(matrix.X[i,j]):
                D['Unknown'].append((i,j))
            elif matrix.X[i,j]==0:
                D['Not_regulated'].append((i,j))
            elif matrix.X[i,j]<0:
                D['Up'].append((i,j))
            elif matrix.X[i,j]>0:
                D['Down'].append((i,j))
            else:
                print(f"{(i,j)} leads to an undefined regualtion state!")
    return(D)


def define_train_and_test_set(S, ratio=0.05):
    D={}
    D['Train']={}
    D['Test']={}

    ratio=0.05

    l=0
    for key in S.keys():
        np.random.seed(l)
        l+=1

        v=S[key]
        np.random.shuffle(v)

        D['Train'][key]=v[:round(len(v)*(1-ratio))]
        D['Test'][key]=v[round(len(v)*(1-ratio)):]
    return(D)



def construct_train_matrix(matrix, D, variant, neg=-10, pos=10):
    l0=len(matrix.obs)
    l1=len(matrix.var)
    
    X=np.zeros((l0,l1))
    
    Dt=D['Train']
    
    for i in D['Train']['Not_regulated']:
        X[i]=neg
    if variant=='up':
        for i in D['Train']['Up']:
            X[i]=pos
        for i in D['Train']['Down']:
            X[i]=neg
    elif variant=='down':
        for i in D['Train']['Down']:
            X[i]=pos
        for i in D['Train']['Up']:
            X[i]=neg
    elif variant=='up+down':
        for i in D['Train']['Up']:
            X[i]=pos
        for i in D['Train']['Down']:
            X[i]=pos
    else:
        print('Variant not defined (must be in "up", "down", "up+down")')
    return(X)


def run_ALS(X, dims=25, regulatization=300, iterations=10**3, alpha=10):
    l0,l1=X.shape
    
    M = sparse.csr_matrix(X.T)
    model = implicit.als.AlternatingLeastSquares(factors=dims, regularization=regulatization, iterations=iterations, random_state=0)
    alpha = 10
    data_conf = (M * alpha).astype('double')
    model.fit(data_conf)
    
    
    C=np.zeros((l0,l1))
    for i in range(l1):
        recommended, recommended_scores = model.recommend(i, data_conf[i], filter_already_liked_items=False, N=l0)

        for j in range(l0):
            C[recommended[j],i]=recommended_scores[j]
    return(C, model)


def evaluate_test_set(D, C, variant, print_hist=True):
    not_reg=[]
    reg=[]

    for i in D['Test']['Not_regulated']:
        not_reg.append(C[i])

    if variant=='up':
        for i in D['Test']['Up']:
            reg.append(C[i])
        for i in D['Test']['Down']:
            not_reg.append(C[i])

    elif variant=='down':
        for i in D['Test']['Down']:
            reg.append(C[i])
        for i in D['Test']['Up']:
            not_reg.append(C[i])

    elif variant=='up+down':
        for i in D['Test']['Down']:
            reg.append(C[i])
        for i in D['Test']['Up']:
            reg.append(C[i])
    
    df_not_reg=pd.DataFrame({'Score':not_reg, 'Set':'Not regulated'})
    df_reg=pd.DataFrame({'Score':reg, 'Set':f'Regulated ({variant})'})
    df=pd.concat([df_not_reg, df_reg])
    df.index=np.arange(len(df))
    
    if print_hist:
        sns.set(rc={'figure.figsize':(12,5)})
        sns.histplot(data=df, x='Score', hue='Set', log_scale=(False, False),bins=51)
        plt.ylim(0, 300)
        
    return(df)


def errors_given_cutoff(df, cut):
    df_not_reg=df[df['Set']=='Not regulated']
    variant=df[df['Set']!='Not regulated']['Set'].iloc[0].split('(')[1][:-1]
    
    df_reg=df[df['Set']==f'Regulated ({variant})']

    false_positives=np.array(df_not_reg['Score']>cut).sum()
    true_negatives=np.array(df_not_reg['Score']<=cut).sum()

    false_negatives=np.array(df_reg['Score']<cut).sum()
    true_positives=np.array(df_reg['Score']>=cut).sum()

    precission=true_positives/(true_positives+false_positives)
    recall=true_positives/(true_positives+false_negatives)

    return(precission, recall)

def precission_recall_curve(df, bins=201, zoom_in=False):
    c=[]
    p=[]
    r=[]

    for cut in np.linspace(-0.5,1.5,bins):

        precission, recall = errors_given_cutoff(df, cut)

        if not np.isnan(precission) and not np.isnan(recall):
            c.append(cut)
            p.append(precission)
            r.append(recall)

    df_precission_recall=pd.DataFrame({'Cutoff':c, 'Precission':p, 'Recall':r, 'Method':'ALS'})
    
    sns.set_style("darkgrid")
    sns.set(rc={'figure.figsize':(9,6)})

    sns.lineplot(data=df_precission_recall, x='Recall', y='Precission', hue='Method')

    if zoom_in:
        plt.ylim(0.7, 1.0)
        plt.xlim(0.7, 1.0)
    else:
        plt.ylim(-0.05, 1.05)
        plt.xlim(-0.05, 1.05)
        
    return(df_precission_recall)

def determine_cutoff(df_precission_recall, error=0.05, method='Precission'):

    l=-1
    for i in range(len(df_precission_recall)-1):
        if 1-error<df_precission_recall[method][i+1] and 1-error>df_precission_recall[method][i]:
            l=i
            break

    if l==-1:
        print(f'It is not possible to chose Recall={1-error}')
    else:
        # Linear interpolation to find the cutoff

        c0=df_precission_recall['Cutoff'][i]
        c1=df_precission_recall['Cutoff'][i+1]

        p0=df_precission_recall[method][i]
        p1=df_precission_recall[method][i+1]

        cut=c0+(c1-c0)*(1-error-p0)/(p1-p0)

    return(cut)

        
# ALS returns some weird gpu.matrix format, this code transforms it into a proper numpy matrix

def gpu_matrix_to_numpy(M):
    Mn=np.zeros(M.shape)
    for i in range(M.shape[0]):
        st=str(M[i])
        st=st[3:-2]
        a=st.split(' ')
        b=[]
        for j in range(len(a)):
            if a[j]==' ' or a[j]=='':
                continue
            else:
                b.append(float(a[j]))
        Mn[i,:]=b
    return(Mn)

def add_ALS_embedding(matrix, model):
    # Adds the embedding of the drugs and the peptides
    matrix.varm['Drug_embedding']=gpu_matrix_to_numpy(model.user_factors)
    matrix.obsm['Peptide_embedding']=gpu_matrix_to_numpy(model.item_factors)
    return(matrix)