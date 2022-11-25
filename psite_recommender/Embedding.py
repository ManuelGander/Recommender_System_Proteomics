import pandas as pd
import numpy as np
import seaborn as sns
import anndata
import scanpy as sc
import scvelo as scv
import plotly.express as px

def colors_based_on_phylogeny(matrix, Path):
    import matplotlib.colors as colors

    Drug_colors=np.load(f'{Path}/Drug_classes_colors.npy', allow_pickle=True)
    Drug_colors=dict(enumerate(Drug_colors.flatten()))[0]

    Emb_matrix=anndata.AnnData(X=matrix.varm['Drug_2d_umap'], obs=matrix.var)
    Emb_matrix.obsm['X_umap']=matrix.varm['Drug_2d_umap']
    Emb_matrix.obs['Drug Class combined']=Emb_matrix.obs['Drug Class combined'].astype('category')


    col=[]
    for i in Emb_matrix.obs['Drug Class combined'].cat.categories:
        col.append(Drug_colors[i])

    col2=[]
    for i in Emb_matrix.obs['Drug Class combined']:
        if not colors.rgb2hex(Drug_colors[i]) in col2:
            col2.append(colors.rgb2hex(Drug_colors[i]))
    return(col, col2)





def calc_umaps(matrix, neighbors_drugs=5, neighbors_peptides=15):
    # 2d drug umap
    Emb_matrix=anndata.AnnData(X=matrix.varm['Drug_embedding'], obs=matrix.var)
    sc.pp.neighbors(Emb_matrix, n_neighbors=neighbors_drugs)
    sc.tl.umap(Emb_matrix)
    matrix.varm['Drug_2d_umap']=Emb_matrix.obsm['X_umap']
    matrix.uns['Drug_emb_2d']=Emb_matrix
    # 3d drug umap
    Emb_matrix=anndata.AnnData(X=matrix.varm['Drug_embedding'], obs=matrix.var)
    sc.pp.neighbors(Emb_matrix, n_neighbors=neighbors_drugs)
    sc.tl.umap(Emb_matrix, n_components=3)
    matrix.varm['Drug_3d_umap']=Emb_matrix.obsm['X_umap']
    matrix.uns['Drug_emb_3d']=Emb_matrix

    # 2d peptide umap
    Emb_matrix=anndata.AnnData(X=matrix.obsm['Peptide_embedding'], obs=matrix.obs)
    sc.pp.neighbors(Emb_matrix, n_neighbors=neighbors_peptides)
    sc.tl.umap(Emb_matrix)
    matrix.obsm['Peptide_2d_umap']=Emb_matrix.obsm['X_umap']
    matrix.uns['Peptide_emb_2d']=Emb_matrix
    # 3d peptide umap
    Emb_matrix=anndata.AnnData(X=matrix.obsm['Peptide_embedding'], obs=matrix.obs)
    sc.pp.neighbors(Emb_matrix, n_neighbors=neighbors_peptides)
    sc.tl.umap(Emb_matrix, n_components=3)
    matrix.obsm['Peptide_3d_umap']=Emb_matrix.obsm['X_umap']
    matrix.uns['Peptide_emb_3d']=Emb_matrix
    
    return(matrix)

def plot_drug_umap(matrix, Path, color_key='Drug Class combined', interactive=True, palette='Phylogeny'):
    sns.set_style("whitegrid")
    if palette=='Phylogeny':
        col, palette_chosen=colors_based_on_phylogeny(matrix, Path)
    else:
        palette_chosen=px.colors.qualitative.Light24
        col=px.colors.qualitative.Light24

    Emb_matrix=anndata.AnnData(X=matrix.varm['Drug_2d_umap'], obs=matrix.var)
    Emb_matrix.obsm['X_umap']=matrix.varm['Drug_2d_umap']
    
    if interactive:
        M=matrix.varm['Drug_2d_umap']
        (x,y)=(M[:,0], M[:,1])


        dfs=pd.DataFrame({'x':x, 'y':y, color_key:matrix.var[color_key], 'Drug Names':matrix.var['Drug Names'],
                         'Drug Class':matrix.var['Drug Class']})

        fig = px.scatter(dfs, x='x', y='y', color=color_key, color_discrete_sequence=palette_chosen,
                               custom_data=[color_key, 'Drug Names', 'Drug Class'])
        fig.update_traces(
            hovertemplate="<br>".join([
                "x: %{x}",
                "y: %{y}",
                "color_key: %{customdata[0]}",
                "Names: %{customdata[1]}",
                "Drug Class: %{customdata[2]}"
            ]))

        fig.update_traces(marker=dict(size=20))

        fig.show()
    else:
        scv.pl.scatter(Emb_matrix, color=color_key, basis='umap', legend_loc='right', palette=col)
        
    
    
def plot_kinobead_umap(matrix, color_key='FGFR1', interactive=True):
    import plotly.express as px
    df=matrix.uns['Drug_class_matrix'].copy()
    df2=matrix.uns['Kinobead_matrix'].copy()
    kinobead_affinity=[a for a in df2[color_key]]
    kinobead_affinity=[0 if np.isnan(a) else a for a in kinobead_affinity]
    log_kinobead_affinity=[0 if a==0 else np.log(a) for a in kinobead_affinity]

    Emb_matrix=anndata.AnnData(X=matrix.varm['Drug_2d_umap'], obs=matrix.var)
    Emb_matrix.obsm['X_umap']=matrix.varm['Drug_2d_umap']
    Emb_matrix.obs[color_key]=log_kinobead_affinity
    
    
    if interactive:

        M=matrix.varm['Drug_2d_umap']
        (x,y)=(M[:,0], M[:,1])


        dfs=pd.DataFrame({'x':x, 'y':y, color_key:log_kinobead_affinity, 'Drug Names':[a for a in matrix.var['Drug Names']],
                         'Drug Class':[a for a in matrix.var['Drug Class']],
                         'Kinobead_affinity':kinobead_affinity})
        fig = px.scatter(dfs, x='x', y='y', color=color_key, color_discrete_sequence=px.colors.qualitative.Light24,
                               custom_data=[color_key, 'Drug Names', 'Drug Class', 'Kinobead_affinity'])
        fig.update_traces(
            hovertemplate="<br>".join([
                "x: %{x}",
                "y: %{y}",
                "color_key: %{customdata[0]}",
                "Names: %{customdata[1]}",
                "Drug Class: %{customdata[2]}",
                "Kinobead_affinity: %{customdata[3]}",
            ]))
        fig.update_traces(marker=dict(size=20))

        fig.show()

    else:
        scv.pl.scatter(Emb_matrix, color=color_key, basis='umap', legend_loc='right')
    
    
def plot_peptide_umap(matrix, color_key='RSRC2', interactive=True):
    import plotly.express as px

    Emb_matrix=anndata.AnnData(X=matrix.obsm['Peptide_2d_umap'], obs=matrix.obs)
    Emb_matrix.obsm['X_umap']=matrix.obsm['Peptide_2d_umap']
    
    df=matrix.uns['KEGG_matrix'].copy()
    Emb_matrix.obs[color_key]=np.array(matrix.obs['Gene Names']==color_key).astype(int)
    
    if interactive:
        M=matrix.obsm['Peptide_2d_umap']
        (x,y)=(M[:,0], M[:,1])

        dfs=pd.DataFrame({'x':x, 'y':y, 
                          color_key:np.array(matrix.obs['Gene Names']==color_key).astype(int), 
                          'Peptide': [a for a in matrix.obs['Modified sequence']],
                          'Gene': [a for a in matrix.obs['Gene Names']],
                          'Reference protein': [a for a in matrix.obs['Reference_protein']],
                          'Ph_Site(s)': [a for a in matrix.obs['Ph_site']]})
        dfs=dfs.sort_values(color_key)

        fig = px.scatter(dfs, x='x', y='y', color=color_key+' '*10,
                        custom_data=[color_key, 'Peptide', 'Gene', 'Reference protein', 'Ph_Site(s)'])

        fig.update_traces(
            hovertemplate="<br>".join([
                "x: %{x}",
                "y: %{y}",
                "color_key: %{customdata[0]}",
                "Peptide: %{customdata[1]}",
                "Gene: %{customdata[2]}",
                "Reference protein: %{customdata[3]}",
                "Ph_Site(s): %{customdata[4]}",
            ]))

        fig.update_traces(marker=dict(size=7))

        fig.show()
    else:
        scv.pl.scatter(Emb_matrix, color=color_key, basis='umap', legend_loc='right', size=100)
    
    
def plot_KEGG_umap(matrix, color_key='Apelin signaling pathway', interactive=True):
    import plotly.express as px

    Emb_matrix=anndata.AnnData(X=matrix.obsm['Peptide_2d_umap'], obs=matrix.obs)
    Emb_matrix.obsm['X_umap']=matrix.obsm['Peptide_2d_umap']
    
    df=matrix.uns['KEGG_matrix'].copy()
    Emb_matrix.obs[color_key]=[a for a in df[color_key]]


    if interactive:    
        M=matrix.obsm['Peptide_2d_umap']
        (x,y)=(M[:,0], M[:,1])

        dfs=pd.DataFrame({'x':x, 'y':y, color_key:Emb_matrix.obs[color_key], 
                          'Peptide': [a for a in matrix.obs['Modified sequence']],
                          'Gene': [a for a in matrix.obs['Gene Names']],
                          'Reference protein': [a for a in matrix.obs['Reference_protein']],
                          'Ph_Site(s)': [a for a in matrix.obs['Ph_site']]})
        dfs=dfs.sort_values(color_key)

        fig = px.scatter(dfs, x='x', y='y', color=color_key,
                        custom_data=[color_key, 'Peptide', 'Gene', 'Reference protein', 'Ph_Site(s)'])

        fig.update_traces(
            hovertemplate="<br>".join([
                "x: %{x}",
                "y: %{y}",
                "color_key: %{customdata[0]}",
                "Peptide: %{customdata[1]}",
                "Gene: %{customdata[2]}",
                "Reference protein: %{customdata[3]}",
                "Ph_Site(s): %{customdata[4]}",
            ]))

        fig.update_traces(marker=dict(size=7))

        fig.show()
    else:
        scv.pl.scatter(Emb_matrix, color=color_key, basis='umap', legend_loc='right', size=100)