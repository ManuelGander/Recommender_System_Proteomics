import pandas as pd
import numpy as np
import seaborn as sns
import anndata

def list_to_str(lst):
    string=''
    for i in lst:
        string+=str(i)+', '
    string=string[:-2]
    return(string)

def ratio_overview(matrix, pr=False):
    n=0
    z=0
    un=0
    up=0
    down=0
    m=len(matrix)*len(matrix.var)

    for i in range(len(matrix)):
        for j in range(len(matrix.var)):
            if matrix.X[i,j]=='Empty':
                n+=1
            elif matrix.X[i,j]=='Unknown':
                un+=1
            elif matrix.X[i,j]=='Not regulated':
                z+=1
            elif matrix.X[i,j]=='Up-regulated':
                up+=1
            elif matrix.X[i,j]=='Down-regulated':
                down+=1
            else:
                print(f'Classification class not found: {matrix.X[i,j]}')
                
                
    if pr:
        print(f'Up-regulated: {round(up/m*10000)/100}%')
        print(f'Down-regulated: {round(down/m*10000)/100}%')
        print(f'Not regulated: {round(z/m*10000)/100}%')
        print(f'Unknown regulation: {round(un/m*10000)/100}%')
        if n>=1:    
            print(f'No curve: {round(n/m*10000)/100}%')
            
    df=pd.DataFrame({'Up-reg.':[up/m], 'Down-reg.':down/m, 'Not reg.':z/m, 'Unknown':un/m, 'No curve':n/m, 'Total amount of regulated p-sites':up+down})
        
    return(df)


def filter_drugs(matrix, min_reg_phos=10):
    # min_reg_phos = min_regulated_psites_per_drug 
    sel=[]
    for i in range(len(matrix.var)):
        k=0
        for j in range(len(matrix)):
            entry=matrix.X[j,i]
            if entry=='Unknown' or entry=='Not regulated' or entry=='Empty':
                continue
            else:
                k+=1
        if k>min_reg_phos-1:
            sel.append(i)
    return(matrix[:,sel])

def filter_peptides(matrix, min_reg_pept=3):
    # min_reg_pept = min_drugs_per_regulated_psite
    sel=[]
    for i in range(len(matrix)):
        k=0
        for j in range(len(matrix.var)):
            entry=matrix.X[i,j]
            if entry=='Unknown' or entry=='Not regulated' or entry=='Empty':
                continue
            else:
                k+=1
        if k>min_reg_pept-1:
            sel.append(i)
    return(matrix[sel,:])


def load_matrix(path, cell_line='A204'):
    Matrix=pd.read_csv(f'{path}', sep='\t', on_bad_lines='skip')

    obs=Matrix[['Gene Names', 'Proteins', 'Modified sequence']]
    obs.index=[str(a) for a in obs.index]
    Matrix=Matrix.drop(['Gene Names', 'Proteins', 'Modified sequence', 'Sequence'], axis=1)
    drugs=[a for a in Matrix.columns]
    X=np.array(Matrix.values, dtype=np.float32)
    n0,n1=X.shape

    Xn=np.full(X.shape, '-'*20, dtype='<U20')
    for i in range(n0):
        for j in range(n1):
            if np.isnan(X[i,j]):
                Xn[i,j]='Unknown'
            elif X[i,j]==0:
                Xn[i,j]='Not regulated'
            elif X[i,j]<0:
                Xn[i,j]='Down-regulated'
            elif X[i,j]>0:
                Xn[i,j]='Up-regulated'

    matrix=anndata.AnnData(X=Xn, obs=obs, dtype=Xn.dtype)
    matrix.var['Drug']=drugs
    return(matrix.copy())


def add_drug_names_and_classes(matrix, Path):
    # Translate TOPAS-ID to drug name and add drug classes
    Drug_name_dict=np.load(f'{Path}/TOPAS_Drugs_A204.npy', allow_pickle=True)
    Drug_name_dict=dict(enumerate(Drug_name_dict.flatten()))[0]

    Drug_class_dict=np.load(f'{Path}/Drug_classes.npy', allow_pickle=True)
    Drug_class_dict=dict(enumerate(Drug_class_dict.flatten()))[0]

    Drug_class_combined_dict=np.load(f'{Path}/Drug_classes_combined.npy', allow_pickle=True)
    Drug_class_combined_dict=dict(enumerate(Drug_class_combined_dict.flatten()))[0]
    
    
    matrix.var['Drug Names']='-'*20
    matrix.var['Drug Class']='-'*20
    matrix.var['Drug Class combined']='-'*20
    
    for i in range(len(matrix.var)):
        drug=matrix.var['Drug'][i]
        matrix.var['Drug Names'][i]=Drug_name_dict[drug]
        matrix.var['Drug Class'][i]=Drug_class_dict[Drug_name_dict[drug]]
        matrix.var['Drug Class combined'][i]=Drug_class_combined_dict[Drug_name_dict[drug]]

    return(matrix.copy())

def add_Kegg_annontations(matrix, Path):
    # Matthew provided me with Kegg pathway annotations, they are added here to the respective peptides
    
    # Load the Dictionaries:

    KEGG_gene=np.load(f'{Path}/KEGG_genes.npy', allow_pickle=True)
    KEGG_gene=dict(enumerate(KEGG_gene.flatten()))[0]
    
    KEGG_protein=np.load(f'{Path}/KEGG_proteins.npy', allow_pickle=True)
    KEGG_protein=dict(enumerate(KEGG_protein.flatten()))[0]
    
    kegg_gene=[['-']] * len(matrix)
    kegg_protein=[['-']] * len(matrix)
    kegg_combinded=[['-']] * len(matrix)

    for i in range(len(matrix)):
        gene=matrix.obs['Gene Names'][i]
        protein=matrix.obs['Proteins'][i]
        
        if gene in KEGG_gene.keys():
            kegg_gene[i]=KEGG_gene[gene]
        else:
            kegg_gene[i]=['-']
            
        if protein in KEGG_protein.keys():
            kegg_protein[i]=KEGG_protein[protein]
        else:
            kegg_protein[i]=['-']
            
        KEGG_comb=[*kegg_gene[i], *kegg_protein[i]]
        KEGG_comb=[a for a in set(KEGG_comb)]
        
        if KEGG_comb=='-':
            kegg_combinded[i]=['-']
        else:
            KEGG_comb.remove('-')
            kegg_combinded[i]=KEGG_comb
            
    matrix.obs['KEGG_gene']=[list_to_str(a) for a in kegg_gene]
    matrix.obs['KEGG_protein']=[list_to_str(a) for a in kegg_protein]
    matrix.obs['KEGG_combined']=[list_to_str(a) for a in kegg_combinded]
        
    return(matrix.copy())


def add_Phosphosite_annotations(matrix, psite_annotation_path):
    
    import sys
    sys.path.append(psite_annotation_path)
    import psite_annotation as pa
    
    df=matrix.obs.copy()    
    
    # Adds effect of phosphorylation
    pa.addPeptideAndPsitePositions(df, pa.pspFastaFile, pspInput = True)
    pa.addPSPAnnotations(df, pa.pspAnnotationFile)
    pa.addPSPRegulatoryAnnotations(df, pa.pspRegulatoryFile)
    pa.addPSPKinaseSubstrateAnnotations(df, pa.pspKinaseSubstrateFile)
    #pa.addDomains(df, pa.domainMappingFile)
    #pa.addMotifs(df, pa.motifsFile)
    pa.addInVitroKinases(df, pa.inVitroKinaseSubstrateMappingFile)
    

    
    matrix.obs=df.copy()
    return(matrix.copy())

def add_Yasushi_annotations(matrix, Path):
    
    Yas_dict=np.load(f'{Path}/Yasushi_annotations.npy', allow_pickle=True)
    Yas_dict=dict(enumerate(Yas_dict.flatten()))[0]

    yasushi=[['-']] * len(matrix)
    
    for i in range(len(matrix)):
        gene=matrix.obs['Gene Names'][i]
        if gene in Yas_dict.keys():
            yasushi[i]=list_to_str(Yas_dict[gene])
        else:
            yasushi[i]='-'
    matrix.obs['Yasushi']=yasushi
    return(matrix)

def upstream(df, key, Path):
    Tr=np.load(f'{Path}/Gene_name_translator.npy', allow_pickle=True)
    Tr=dict(enumerate(Tr.flatten()))[0]
    
    if key=='vivo':
        key2='PSP'
    if key=='vitro':
        key2='In Vitro'
    
    df[f'Upstream_{key}']=[a for a in df[f'{key2} Kinases']]
    upstream=['']*len(df)
    
    for i in range(len(df)):
        s=df[f'{key2} Kinases'][i]
        ls=s.split(';')

        if len(s)>1:
            for j in range(len(ls)):
                st=ls[j]
                if '/' in st:
                    st=st.split('/')[0]
                if '[' in st:
                    st=st.split('[')[0]
                if st in Tr.keys():
                    st=Tr[st]
                upstream[i]+=st+', '

    up2=[a[:-2] for a in upstream]

    up2_new=np.full(len(up2), '', dtype='<100U')
    for i in range(len(up2)):
        for a in set(up2[i].split(', ')):
            if a!='':
                up2_new[i]+=', '+a
        if up2_new[i]!='':
            up2_new[i]=up2_new[i][2:]
                
                
    df[f'Upstream_{key}']=up2_new
      
    return(df)

def add_up_and_downstream_annotations(matrix, Path):
    df=matrix.obs.copy()
    
    
    # Adds downstream annotation
    
    down_act=[''] * len(matrix)
    down_inh=[''] * len(matrix)
    down_unk=[''] * len(matrix)

    for i in range(len(matrix)):

        s=df['PSP_ON_PROT_INTERACT'][i]

        if s!='-':
            ls=df['PSP_ON_PROT_INTERACT'][i].split('; ')

            for j in range(len(ls)):

                if ls[j][-9:]=='(INDUCES)':
                    if down_act[i]!='':
                        down_act[i]+=', '
                    down_act[i]+=ls[j][:-9]
                elif ls[j][-10:]=='(DISRUPTS)':
                    if down_inh[i]!='':
                        down_inh[i]+=', '
                    down_inh[i]+=ls[j][:-10]
                elif ls[j][-14:]=='(NOT_REPORTED)':
                    if down_unk[i]!='':
                        down_unk[i]+=', '
                    down_unk[i]+=ls[j][:-14]
                    
    # Adds upstream annotation
                    
    df=upstream(df, 'vivo', Path)
    df=upstream(df, 'vitro', Path)
    
    df['Downstream_activated']=down_act
    df['Downstream_inhibited']=down_inh
    df['Downstream_unknown']=down_unk
    
    matrix.obs=df.copy()
    
    return(matrix.copy())

def apply_basic_filtering(matrix):
    matrix=filter_drugs(matrix).copy()
    matrix=filter_peptides(matrix).copy()
    return(matrix)


def create_drug_class_matrix(Path, matrix, cutoff=1000):
    kinobead_data=pd.read_csv(f'{Path}/drug_target_matrix.tsv', sep='\t')
    drugs=kinobead_data['Drug']
    kinobead_data.index=drugs
    kinobead_data=kinobead_data.drop(['Drug','Unnamed: 1'], axis=1)

    df=kinobead_data.copy()

    n0,n1=df.shape

    bools=df.values>cutoff
    bools=np.array(bools, dtype=int)

    dfn=pd.DataFrame(bools, columns = df.columns)
    dfn.index=kinobead_data.index

    dfn=dfn.T[matrix.var['Drug Names']].T.copy()
    return(dfn, kinobead_data.T[dfn.index].T)

def create_KEGG_matrix(matrix):
    # Get all classes of KEGG_combined
    Dx={}
    for i in range(len(matrix)):
        KEGG_comb=matrix.obs['KEGG_combined'][i]

        if KEGG_comb!='':
            elements=matrix.obs['KEGG_combined'][i].split(',')        
            for e in elements:
                if e!='':
                    if e[0]==' ':
                        e=e[1:]
                if not e in Dx.keys():
                    Dx[e]=len(Dx)

    KEGG_matrix=np.zeros((len(matrix),len(Dx)), dtype=int)

    for i in range(len(matrix)):
        KEGG_comb=matrix.obs['KEGG_combined'][i]

        if KEGG_comb!='':
            elements=matrix.obs['KEGG_combined'][i].split(',')        
            for e in elements:
                if e!='':
                    if e[0]==' ':
                        e=e[1:]
                if e in Dx.keys():
                    KEGG_matrix[i,Dx[e]]=1
                else:
                    print(e)  
    df = pd.DataFrame(KEGG_matrix, columns = [a for a in Dx.keys()])
    df.index=matrix.obs['Modified sequence']
    return(df)


def cleanup(matrix):
    df=matrix.obs.copy()
    for i in ['Start positions', 'End positions', 'Site sequence context', 'Site positions', 'PSP_LT_LIT', 
              'PSP_MS_LIT', 'PSP_MS_CST', 'PSP_ON_FUNCTION', 'PSP_ON_PROCESS', 'PSP_ON_PROT_INTERACT', 
              'PSP_ON_OTHER_INTERACT', 'PSP_NOTES', 'PSP Kinases', 'In Vitro Kinases']:
        del df[i]
        
    for i in ['KEGG_gene', 'KEGG_protein', 'KEGG_combined', 'Yasushi', 'Downstream_activated', 
              'Downstream_inhibited', 'Downstream_unknown', 'Upstream_vivo', 'Upstream_vitro']:
        arr=[''*200]*len(matrix)
        for j in range(len(matrix)):
            if df[i][j]=='':
                arr[j]='-'
            else:
                arr[j]=df[i][j]
        df[i]=arr
    matrix.obs=df.copy()
    
    matrix.var.index=np.arange(len(matrix.var))
    matrix.obs.index=np.arange(len(matrix))
    
    matrix.obs.index=[str(a) for a in matrix.obs.index]
    matrix.var.index=[str(a) for a in matrix.var.index]
    
    return(matrix.copy())

def colors_based_on_phylogeny(matrix, Path):
    import matplotlib.colors as colors

    Drug_colors=np.load(f'{Path}/Drug_classes_colors.npy', allow_pickle=True)
    Drug_colors=dict(enumerate(Drug_colors.flatten()))[0]

    Emb_matrix=anndata.AnnData(X=matrix.varm['Drug_2d_umap'], obs=matrix.var)
    Emb_matrix.obsm['X_umap']=matrix.varm['Drug_2d_umap']


    col=[]
    for i in Emb_matrix.obs['Drug Class combined'].cat.categories:
        col.append(Drug_colors[i])

    col2=[]
    for i in Emb_matrix.obs['Drug Class combined']:
        if not colors.rgb2hex(Drug_colors[i]) in col2:
            col2.append(colors.rgb2hex(Drug_colors[i]))
    return(col, col2)



def plot_summary(before_filtering, after_filtering):
    import matplotlib.pyplot as plt
    import numpy as np
    
    before_filtering['When']='Before filtering'
    after_filtering['When']='After filtering'
    comb=pd.concat([before_filtering, after_filtering])
    comb.index=[0,1]

    classes=comb.columns[:5]

    df0=pd.DataFrame({'Percentage': before_filtering.values[0][:5]*100, 'Classes': classes, 'When':'Before filtering'})
    df1=pd.DataFrame({'Percentage': after_filtering.values[0][:5]*100, 'Classes': classes, 'When':'After filtering'})

    DF=pd.concat([df0, df1])
    DF.index=np.arange(len(DF))


    sns.set_style('darkgrid')

    fig, axs = plt.subplots(1,2,figsize=(10,4), gridspec_kw={'width_ratios': [1, 2]})

    #### Subplot 1
    plt.subplot(1, 2, 1)

    sns.barplot(data=comb, x='When', y='Total amount of regulated p-sites')

    #### Subplot 2
    plt.subplot(1, 2, 2)
    sns.barplot(data=DF, x='Classes', y='Percentage', hue='When')

    plt.show()
    
def add_kinobead_and_kegg_matrices(matrix, Path):
    matrix.uns['Drug_class_matrix'], matrix.uns['Kinobead_matrix']=create_drug_class_matrix(Path, matrix, cutoff=1000)
    matrix.uns['KEGG_matrix']=create_KEGG_matrix(matrix)
    return(matrix)




def minimize_site_positions(matrix):
    df=matrix.obs.copy()
    df['Index']=np.arange(len(df))
    df['Gene Names']=df['Gene Names'].astype('category')
    genes=df['Gene Names'].cat.categories

    frames=[]
    for gene in genes:
        dfg=df[df['Gene Names']==gene].copy()
        dfg=reduce_site_positions(dfg)
        frames.append(dfg)

    dfc=pd.concat(frames)
    dfc=dfc.sort_values('Index').copy()

    matrix.obs['Reference_protein']=dfc['Reference_protein']
    matrix.obs['Ph_site']=dfc['Ph_site']
    
    return(matrix)


def identify_most_common_protein(dfg):
    prots=[]
    for i in range(len(dfg)):
        sites=dfg['Site positions'][i].split(',')
        for site in sites:
            prots.append(site.split('_')[0])
    prots=[a for a in set(prots)]

    if prots==['']:
        most_common='-'

    else:
        D={}
        for prot in prots:
            if prot=='':
                continue
            else:
                z=np.zeros(len(dfg))
                for i in range(len(dfg)):
                    if prot in dfg['Site positions'][i]:
                        z[i]=1
                D[prot]=[z.sum()]
        dtf=pd.DataFrame(D).T

        most_common=dtf.sort_values(0).index[-1]

    return(most_common)

def reduce_site_positions(dfg):
    com=identify_most_common_protein(dfg)

    site_pos=[]
    for i in range(len(dfg)):
        entry='-'
        sites=dfg['Site positions'][i].split(';')
        for site in sites:
            if com in site:
                if entry=='-':
                    entry=site.split('_')[1]
                else:
                    entry+='_'+site.split('_')[1]
        site_pos.append(entry)
    dfg['Reference_protein']=com
    dfg['Ph_site']=site_pos
    return(dfg)

    
    
    
def preprocess(matrix, Path, psite_annotation_path, min_reg_phos=10, min_reg_pept=3):
    matrix=add_drug_names_and_classes(matrix, Path)
    matrix=add_Kegg_annontations(matrix, Path)
    matrix=add_Phosphosite_annotations(matrix, psite_annotation_path).copy()
    matrix=minimize_site_positions(matrix)
    matrix=add_Yasushi_annotations(matrix, Path)
    matrix=add_up_and_downstream_annotations(matrix, Path)
    before_filtering=ratio_overview(matrix, pr=False)
    matrix=filter_drugs(matrix, min_reg_phos).copy()
    matrix=filter_peptides(matrix, min_reg_pept).copy()
    after_filtering=ratio_overview(matrix, pr=False)
    plot_summary(before_filtering, after_filtering)
    matrix=cleanup(matrix.copy())
    matrix=add_kinobead_and_kegg_matrices(matrix, Path)
    return(matrix)
    