import pandas as pd
import numpy as np
import seaborn as sns
import anndata
import scanpy as sc
import scvelo as scv
import statsmodels.stats.multitest
import scipy
import matplotlib.pyplot as plt


def calculate_leiden_peptides(matrix, resolution=2, plot=True):
    Emb=matrix.uns['Peptide_emb_2d']
    sc.pp.neighbors(Emb)
    sc.tl.leiden(Emb, resolution=resolution)
    
    if plot:
        scv.pl.scatter(Emb, color='leiden', size=50, legend_loc='right', palette=sns.color_palette('tab20'))   
    matrix.obs['leiden']=Emb.obs['leiden']
    return(matrix)


def calculate_leiden_drugs(matrix, resolution=2, plot=True):
    Emb=matrix.uns['Drug_emb_2d']
    sc.pp.neighbors(Emb)
    sc.tl.leiden(Emb, resolution=resolution)
    
    if plot:
        scv.pl.scatter(Emb, color='leiden', legend_loc='right', palette=sns.color_palette('tab20'))   
    matrix.var['leiden']=Emb.obs['leiden']
    return(matrix)

def most_ALS_extreme_peptides(matrix, k=500, plot_hist=False):
    Emb=matrix.obsm['Peptide_embedding']

    D={}
    for i in range(Emb.shape[1]):
        emb_i=Emb[:,i]
        D[i]=np.argpartition(emb_i,-k)[-k:]


        if plot_hist:
            mn=min(emb_i[D[i]])        
            sns.histplot(x=emb_i, bins=101)

            plt.plot([mn,mn], [0,1000], color='brown', marker='o', linestyle='dashed', linewidth=2, markersize=5)
            plt.pause(0.01)
    return(D)

def most_ALS_extreme_drugs(matrix, k=20, plot_hist=False):
    Emb=matrix.varm['Drug_embedding']

    D={}
    for i in range(Emb.shape[1]):
        emb_i=Emb[:,i]
        D[i]=np.argpartition(emb_i,-k)[-k:]


        if plot_hist:
            mn=min(emb_i[D[i]])        
            sns.histplot(x=emb_i, bins=101)

            plt.plot([mn,mn], [0,1000], color='brown', marker='o', linestyle='dashed', linewidth=2, markersize=5)
            plt.pause(0.01)
    return(D)


def leiden_peptides(matrix):
    
    matrix.obs['leiden']=matrix.obs['leiden'].astype('category')
    df=matrix.obs['leiden']
    leidens=[a for a in df.cat.categories]

    D={}
    for i in range(len(leidens)):
        D[i]=np.where(df==leidens[i])[0]

    return(D)


def leiden_drugs(matrix):
    
    matrix.var['leiden']=matrix.var['leiden'].astype('category')
    df=matrix.var['leiden']
    leidens=[a for a in df.cat.categories]

    D={}
    for i in range(len(leidens)):
        D[i]=np.where(df==leidens[i])[0]

    return(D)

def load_kegg_matrix(matrix, signaling_only=True):
    df=matrix.uns['KEGG_matrix']
    
    if signaling_only:
        # Keep only pathways that contain "signal" in it's name

        keep=[]

        for i in range(len(df.T)):
            clas=df.columns[i]
            if 'signal' in clas:
                keep.append(clas)
        df=df[keep].copy()
        
    return(df)

def load_drug_class_matrix(matrix):
    df=matrix.uns['Drug_class_matrix']
    dfs=df.T[df.sum()>1].T
        
    return(dfs)



def binomial_test(D, df, mulititest=True):
    Base_prob={}
    for i in range(len(df.T)):
        clas=df.columns[i]
        Base_prob[clas]=df.sum()[i]/len(df)


    P=np.zeros((len(D), df.shape[1]))
    for i in range(len(D)):
        occurrences=df.iloc[D[i]].sum()
        for j in range(len(df.columns)):
            clas=[a for a in df.columns][j]             
            found=occurrences[clas]
            base_prob=Base_prob[clas]

            P[i,j]=scipy.stats.binomtest(found, len(D[i]), p=base_prob, alternative='greater').pvalue


    p=P.flatten()
    q=statsmodels.stats.multitest.multipletests(p, alpha=0.05, method='fdr_bh')[1]
    Q=q.reshape(P.shape)

    if mulititest:
        Pdf=pd.DataFrame(Q)
    else:
        Pdf=pd.DataFrame(P)
        
    Pdf.index=[a for a in np.arange(len(D.keys()))]
    Pdf.columns=df.columns
    return(Pdf)


def calc_clustermap(Pdf):
    cluster_map=sns.clustermap(Pdf, figsize=(0.0001, 0.0001), cmap='magma_r', vmin=0, vmax=0.1)
    plt.axis('off')
    cluster_map._figure=None
    col_reorder=cluster_map.dendrogram_col.reordered_ind
    row_reorder=cluster_map.dendrogram_row.reordered_ind
    
    return(col_reorder, row_reorder)



def print_sub_heatmap(Pdf, col_reorder, row_reorder, i0, i1=0):
    if i1==0:
        i1=i0+50
    cols=[[a for a in Pdf.columns][i] for i in col_reorder]
    rows=[[a for a in Pdf.index][i] for i in row_reorder]
    
    Pdfn=Pdf.copy()
    
    Pdfn=Pdfn[cols]
    Pdfn=Pdfn.T
    Pdfn=Pdfn[rows]
    Pdfn=Pdfn.T
    Pdfn.index=[str(i) for i in Pdfn.index]
    
    plt.figure(figsize = (20,10), dpi=100)
    sns.heatmap(Pdfn.T[i0:i1].T, cmap='magma_r', vmin=0, vmax=0.1)
    
    
def cut_of_p_values(Pdf, cutoff=0.1):
    x,y=np.where(Pdf>cutoff)
    for i in range(len(x)):
        Pdf.values[x[i],y[i]]=1
    Pdf=Pdf.T[Pdf.sum()<Pdf.shape[0]-0.92].T
    return(Pdf)




def calc_peptide_heatmap(matrix, clustering='Leiden', signaling_only=True, cutoff=0.1):
    if clustering=='ALS':
        D=most_ALS_extreme_peptides(matrix)
    elif clustering=='Leiden':
        D=leiden_peptides(matrix)
    else:
        print(f'Error: Unknown clustering (not in (ALS, Leiden))')
    df=load_kegg_matrix(matrix, signaling_only=signaling_only)
    Pdf=binomial_test(D,df)
    Pdf=cut_of_p_values(Pdf, cutoff=cutoff)
    col_reorder, row_reorder=calc_clustermap(Pdf)
    print_sub_heatmap(Pdf, col_reorder, row_reorder, i0=0, i1=200)
    
    return(Pdf)


def calc_drug_heatmap(matrix, clustering='Leiden', cutoff=0.1):
    if clustering=='ALS':
        D=most_ALS_extreme_drugs(matrix)
    elif clustering=='Leiden':
        D=leiden_drugs(matrix)
    else:
        print(f'Error: Unknown clustering (not in (ALS, Leiden))')
    df=load_drug_class_matrix(matrix)
    Pdf=binomial_test(D,df, mulititest=False)
    Pdf=cut_of_p_values(Pdf, cutoff=cutoff)
    col_reorder, row_reorder=calc_clustermap(Pdf)
    print_sub_heatmap(Pdf, col_reorder, row_reorder, i0=0, i1=200)
    
    return(Pdf)

    