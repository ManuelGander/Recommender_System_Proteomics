import pandas as pd
import numpy as np
import seaborn as sns
import anndata
import matplotlib.pyplot as plt
import scipy


def run_ALS(X, dims=25, regulatization=300, iterations=10**3, alpha=10):
    import implicit
    from scipy import sparse
    n0,n1=X.shape
    
    M = sparse.csr_matrix(X.T)
    model = implicit.als.AlternatingLeastSquares(factors=dims, regularization=regulatization, iterations=iterations, random_state=0)
    alpha = alpha
    data_conf = (M * alpha).astype('double')
    model.fit(data_conf)
    
    
    C=np.zeros((n0,n1))
    for i in range(n1):
        recommended, recommended_scores = model.recommend(i, data_conf[i], filter_already_liked_items=False, N=n0)

        for j in range(n0):
            C[recommended[j],i]=recommended_scores[j]
            
            
    X0=model.item_factors
    Y0=model.user_factors
    X0=gpu_matrix_to_numpy(X0)
    Y0=gpu_matrix_to_numpy(Y0)

    return(C, X0, Y0)

def gpu_matrix_to_numpy(M):
    n0,n1=M.shape

    Dt={}
    for i in range(n0):
        Dt[i]=[]
        st=str(M[i])
        st=st[2:-2]
        a=st.split(' ')

        for e in a:
            Dt[i].append(e.split('\n')[0])
    Dtn={}
    for i in range(n0):
        lst=[]
        for e in Dt[i]:
            if e!='' and e!=' ':
                lst.append(float(e))
        Dtn[i]=np.array(lst)

    Mn=np.zeros(M.shape)
    for i in range(n0):
        if len(Dtn[i])!=n1:
            print(f'Error in gpu_matrix to numpy matrix conversion in row {i}!')
            print(Dtn[i])
        else:
            Mn[i]=Dtn[i]
    return(Mn)

def construct_weight_matrix(matrix, D, key=None):
    n0,n1=matrix.shape
    
    Dt=D['Train']
    train_set=[*Dt['Not regulated'], *Dt['Up-regulated'], *Dt['Down-regulated']]
    
    if key!='pEC50':
        W=np.full((n0,n1), 0.1)
        for i in train_set:
            W[i]=1
    else:
        for i in train_set:
            res=1+(float(matrix.layers['pEC50'][i])/float(matrix.layers['Log EC50 Error'][i]))
            if res<1000:
                W[i]=res
            else:
                W[i]=1000
    return(W)



def run_ALS_with_weights(matrix, D, X, dims=25, regulatization=10, iterations=10**2, weight_empty=0.01, weight_unknown=0.01, key=None):
    n0,n1=X.shape
    
    # Construct weights-matrix
    W=construct_weight_matrix(matrix, D, key=key)
                
    X0,Y0=run_weighted_als(X,W,dims,iterations,regulatization)
        
    C=np.matmul(X0.T,Y0)
    
    return(C,X0,Y0)


def run_weighted_als(X,W,l,iterations,regulatization):
    n0,n1=X.shape   
    
    X1=np.zeros((l,n0))
    Y0=np.random.rand(l,n1)
    Y1=np.zeros((l,n1))
    
    for k in range(iterations):
        for i in range(n0):

            cd=W[i,:]
            pd=X[i,:]

            cY0T=Y0.T * cd[:, np.newaxis]
            M=np.matmul(Y0,cY0T)+regulatization*np.diag(np.ones(l))
            b=np.matmul(Y0,pd*cd)
            X1[:,i]=np.linalg.solve(M,b)
        X0=X1.copy()

        for i in range(n1):

            cp=W[:,i]
            pp=X[:,i]
            cX0T=X0.T * cp[:, np.newaxis]
            M=np.matmul(X0,cX0T)+regulatization*np.diag(np.ones(l))

            b=np.matmul(X0,pp*cp)
            Y1[:,i]=np.linalg.solve(M,b)
        Y0=Y1.copy()
        
    return(X0,Y0)




def calc_correlation(matrix, D, C, key, variant, set_used):
    set_used_copy=set_used
    x=[]
    y=[]
    
    if set_used=='up':
        set_used=D['Test']['Up-regulated']
    elif set_used=='down':
        set_used=D['Test']['Down-regulated']
    elif set_used=='up+down':
        set_used=[*D['Test']['Up-regulated'], *D['Test']['Down-regulated']]
    elif set_used=='unknown':
        set_used=D['Test']['Unknown']
    elif set_used=='not':
        set_used=D['Test']['Not regulated']
    elif set_used=='empty':
        set_used=D['Test']['Empty']
    elif set_used=='up+down+not':
        set_used=[*D['Test']['Up-regulated'], *D['Test']['Down-regulated'], *D['Test']['Not regulated']]
    elif set_used=='up+down+not+unknown':
        set_used=[*D['Test']['Up-regulated'], *D['Test']['Down-regulated'], *D['Test']['Not regulated'], *D['Test']['Unknown']]
    else:
        print(f'Error: set_used {key} is not in [up, down, not, unknown, empty, up+down, up+down+not, up+down+not+unknown]')
    
    
    if 'empty' in set_used_copy:
        for i in set_used:
            y.append(C[i])
            
        sns.histplot(y)
        return((-999,-999),(-999,-999))
    else:
        for i in set_used:
            if key in ['pEC50', 'log(Fold_change_predicted)']:
                x.append(float(matrix.layers[key][i]))
            else:
                x.append(np.log(float(matrix.layers[key][i])))
            y.append(C[i])

        pearson=scipy.stats.pearsonr(x,y)
        spearman=scipy.stats.spearmanr(x,y)
        print(f'Spearman correlation: {spearman[0]}, p-value: {spearman[1]}')
        print(f'Pearson correlation: {pearson[0]}, p-value: {pearson[1]}')

        sns.kdeplot(x=x[:10**4], y=y[:10**4], fill=True)

        return(pearson, spearman)


def determine_cutoff(df_precission_recall, error, method):

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

def determine_cutoffs(df_precission_recall, method='Precission'):
    Dict={}
    for error in [0.5,0.1,0.05,0.01,0.001]:
        if method=='Precission':
            Dict[f'Type 1 error: {error}']=determine_cutoff(df_precission_recall, error=error, method=method)
        elif method=='Recall':
            Dict[f'Type 2 error: {error}']=determine_cutoff(df_precission_recall, error=error, method=method)
        else:
            print('ERROR: method {method} is not in [Precission, Recall]')
    return(Dict)




def fast_error_to_cutoff(df_not_reg, df_reg, cut):
    false_positives=np.array(df_not_reg['Score']>cut).sum()
    true_positives=np.array(df_reg['Score']>=cut).sum()
    precission=true_positives/(true_positives+false_positives)
    return(precission)


def score_to_confidence(matrix, df):
    l0=len(matrix.obs)
    l1=len(matrix.var)
    
    df_not_reg=df[df['Set']=='Not regulated']
    variant=df[df['Set']!='Not regulated']['Set'].iloc[0].split('(')[1][:-1]
    
    df_reg=df[df['Set']==f'Regulated ({variant})']
    
    C=matrix.layers['ALS_scores']
    
    Confidence=np.zeros((l0,l1))
    for i in range(l0):
        for j in range(l1):
            Confidence[i,j]=fast_error_to_cutoff(df_not_reg, df_reg, C[i,j])
    matrix.layers['Confidence']=Confidence
    return(matrix)



def add_ALS_embedding(matrix, model):
    # Adds the embedding of the drugs and the peptides
    matrix.varm['Drug_embedding']=gpu_matrix_to_numpy(model.user_factors)
    matrix.obsm['Peptide_embedding']=gpu_matrix_to_numpy(model.item_factors)
    return(matrix)