import pandas as pd
import numpy as np
import seaborn as sns
import anndata
import scanpy as sc
import scvelo as scv
import statsmodels.stats.multitest



def infer_selection_type(matrix, selection):
    # Try to infer selection type:
    if selection[0] in [a for a in matrix.var['Drug']]:
        selection_type='Drug'
    elif selection[0] in [a for a in matrix.obs['Gene']]:
        selection_type='Gene'
    elif selection[0] in [a for a in matrix.obs['Protein']]:
        selection_type='Protein'
    return(selection_type)



def calc_umaps(matrix):
    # 2d drug umap
    Emb_matrix=anndata.AnnData(X=matrix.varm['Drug_embedding'], obs=matrix.var)
    sc.pp.neighbors(Emb_matrix)
    sc.tl.umap(Emb_matrix)
    matrix.varm['Drug_2d_umap']=Emb_matrix.obsm['X_umap']
    # 3d drug umap
    Emb_matrix=anndata.AnnData(X=matrix.varm['Drug_embedding'], obs=matrix.var)
    sc.pp.neighbors(Emb_matrix)
    sc.tl.umap(Emb_matrix, n_components=3)
    matrix.varm['Drug_3d_umap']=Emb_matrix.obsm['X_umap']

    # 2d peptide umap
    Emb_matrix=anndata.AnnData(X=matrix.obsm['Peptide_embedding'], obs=matrix.obs)
    sc.pp.neighbors(Emb_matrix)
    sc.tl.umap(Emb_matrix)
    matrix.obsm['Peptide_2d_umap']=Emb_matrix.obsm['X_umap']
    # 3d peptide umap
    Emb_matrix=anndata.AnnData(X=matrix.obsm['Peptide_embedding'], obs=matrix.obs)
    sc.pp.neighbors(Emb_matrix)
    sc.tl.umap(Emb_matrix, n_components=3)
    matrix.obsm['Peptide_3d_umap']=Emb_matrix.obsm['X_umap']
    
    return(matrix)

def plot_2d_umap(matrix, key, leiden_resolution=0.5, selection=[], selection_type=None, leiden=False):

    if key=='Drug':
        Emb_matrix=anndata.AnnData(X=matrix.varm['Drug_2d_umap'], obs=matrix.var)
        Emb_matrix.obsm['X_umap']=matrix.varm['Drug_2d_umap']
    elif key=='Peptide':
        Emb_matrix=anndata.AnnData(X=matrix.obsm['Peptide_2d_umap'], obs=matrix.obs)
        Emb_matrix.obsm['X_umap']=matrix.obsm['Peptide_2d_umap']
    else:
        print(f'Error - Unknown key: {key} is not in [Drug, Peptide]')

    if leiden==True:
        scv.pl.scatter(Emb_matrix, basis='X_umap', size=200, color='leiden', figsize=(6,5), legend_loc='right')
    else:
        if selection==[] and selection_type==None:
            print('No selection_type specified (choose e.g. Drug, Gene, ...)')
        elif selection==[]:
            Emb_matrix.obs[selection_type]=Emb_matrix.obs[selection_type].astype('category')
            scv.pl.scatter(Emb_matrix, basis='X_umap', size=200, color=selection_type, figsize=(6,5), legend_loc='right')
        else:
            if selection_type==None:
                # Try to infer selection type:
                print('Infering selection type')
                selection_type=infer_selection_type(matrix, selection)
            else:
                for i in selection:
                    if not i in [a for a in Emb_matrix.obs[selection_type]]:
                        print(f'{i} not found in column {selection_type} (typo?)')
            Emb_matrix.obs[selection_type]=Emb_matrix.obs[selection_type].astype('category')
            scv.pl.scatter(Emb_matrix, basis='X_umap', size=200, color=selection_type, groups=selection,
                           figsize=(6,5), legend_loc='right')
            
            
def plot_3d_umap(matrix, key, leiden_resolution=0.5, selection=[], selection_type=None, leiden=False):
    import plotly.express as px
    
    if key=='Drug':
        Emb_matrix=anndata.AnnData(X=matrix.varm['Drug_2d_umap'], obs=matrix.var)
        Ms=matrix.varm['Drug_3d_umap']
    elif key=='Peptide':
        Emb_matrix=anndata.AnnData(X=matrix.obsm['Peptide_2d_umap'], obs=matrix.obs)
        Ms=matrix.obsm['Peptide_3d_umap']
    else:
        print(f'Error - Unknown key: {key} is not in [Drug, Peptide]')

    (x,y,z)=(Ms[:,0], Ms[:,1], Ms[:,2])
        
        
    if selection==[]:
        dfs=pd.DataFrame({'x':x, 'y':y, 'z':z, key:[a for a in Emb_matrix.obs[selection_type]]})

    else:       
        if selection_type==None:
            # Try to infer selection type:
            selection_type=infer_selection_type(matrix, selection)
            print(f'Infered selection type: {selection_type}')
        else:
            for i in selection:
                if not i in [a for a in Emb_matrix.obs[selection_type]]:
                    print(f'{i} not found in column {selection_type} (typo?)')
        
        sel=[]
        for element in Emb_matrix.obs[selection_type]:
            if element in selection:
                sel.append(element)
            else:
                sel.append('not selected')
        dfs=pd.DataFrame({'x':x, 'y':y, 'z':z, key:sel})
        
        
    fig = px.scatter_3d(dfs, x='x', y='y', z='z', color=key)
    fig.show()





