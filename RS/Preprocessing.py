import pandas as pd
import numpy as np
import seaborn as sns
import anndata


import sys
sys.path.append('/home/mgander/Github_repos/psite_annotation-main')



def list_to_str(lst):
    string=''
    for i in lst:
        string+=str(i)+', '
    string=string[:-2]
    return(string)

def ratio_overview(matrix, pr=False):
    n=0
    z=0
    p=0
    m=len(matrix)*len(matrix.var)

    for i in range(len(matrix)):
        for j in range(len(matrix.var)):
            if np.isnan(matrix.X[i,j]):
                n+=1
            elif matrix.X[i,j]==0:
                z+=1
            elif abs(matrix.X[i,j])>0:
                p+=1
    if pr:
        print(f'Regulated: {n/m*100}%')
        print(f'Not-regulated: {z/m*100}%')
        print(f'Unknown regulation: {p/m*100}%')
        
    return(n/m, z/m, p/m, m)


def filter_drugs(matrix, min_reg_phos=10):
    # min_reg_phos = min_regulated_psites_per_drug 
    sel=[]
    for i in range(len(matrix.var)):
        k=0
        for j in range(len(matrix)):
            if np.isnan(matrix.X[j,i]) or matrix.X[j,i]==0:
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
            if np.isnan(matrix.X[i,j]) or matrix.X[i,j]==0:
                continue
            else:
                k+=1
        if k>min_reg_pept-1:
            sel.append(i)
    return(matrix[sel,:])


def load_matrix(path, cell_line='A204'):
    Matrix=pd.read_csv(f'{path}', sep='\t', on_bad_lines='skip')

    # Extract relevant information:
    Matrix['Gene']=Matrix['Gene Names']
    Matrix['Protein']=Matrix['Proteins']

    obs=Matrix[['Gene', 'Protein', 'Modified sequence']]
    obs.index=[str(a) for a in obs.index]
    Matrix=Matrix.drop(['Gene Names', 'Proteins', 'Modified sequence', 'Sequence', 'Gene', 'Protein'], axis=1)
    drugs=[a for a in Matrix.columns]
    X=np.array(Matrix.values, dtype=np.float32)

    matrix=anndata.AnnData(X=X, obs=obs)
    matrix.var['TOPAS-ID']=drugs
    return(matrix)


def add_drug_names(matrix, Path):
    # Translate TOPAS-ID to drug name:
    Drug_name_dict=np.load(f'{Path}/TOPAS_ID_A204.npy', allow_pickle=True)
    Drug_name_dict=dict(enumerate(Drug_name_dict.flatten()))[0]

    matrix.var['Drug']='---------------------------------------------------------------------------'
    for i in range(len(matrix.var)):
        dr=matrix.var['TOPAS-ID'].iloc[i]
        matrix.var['Drug'][i]=Drug_name_dict[f'{dr}']
    return(matrix)

def add_Kegg_annontations(matrix, Path):
    # Matthew provided me with Kegg pathway annotations, they are added here to the respective peptides
    
    
    # Load the Dictionaries:

    KEGG_gene=np.load(f'{Path}/KEGG_genes.npy', allow_pickle=True)
    KEGG_gene=dict(enumerate(KEGG_gene.flatten()))[0]
    
    KEGG_protein=np.load(f'{Path}/KEGG_proteins.npy', allow_pickle=True)
    KEGG_protein=dict(enumerate(KEGG_protein.flatten()))[0]
    
    # 
    
    matrix.obs['KEGG_gene']='-'*200
    matrix.obs['KEGG_protein']='-'*200
    matrix.obs['KEGG_combined']='-'*200
    
    for i in range(len(matrix)):
        gene=matrix.obs['Gene'][i]
        protein=matrix.obs['Protein'][i]
        
        if gene in KEGG_gene.keys():
            matrix.obs['KEGG_gene'][i]=KEGG_gene[gene]
        else:
            matrix.obs['KEGG_gene'][i]=['-']
            
            
        if protein in KEGG_protein.keys():
            matrix.obs['KEGG_protein'][i]=KEGG_protein[protein]
        else:
            matrix.obs['KEGG_protein'][i]=['-']
            
        KEGG_comb=[*matrix.obs['KEGG_gene'][i], *matrix.obs['KEGG_protein'][i]]
        KEGG_comb=[a for a in set(KEGG_comb)]
        
        if KEGG_comb=='-':
            matrix.obs['KEGG_combined'][i]=['-']
        else:
            KEGG_comb.remove('-')
            matrix.obs['KEGG_combined'][i]=KEGG_comb
            
    matrix.obs['KEGG_gene']=[list_to_str(a) for a in matrix.obs['KEGG_gene']]
    matrix.obs['KEGG_protein']=[list_to_str(a) for a in matrix.obs['KEGG_protein']]
    matrix.obs['KEGG_combined']=[list_to_str(a) for a in matrix.obs['KEGG_combined']]
        
    return(matrix)

def add_Phosphosite_annotations(matrix):
    import psite_annotation as pa
    
    df=matrix.obs.copy()
    df['Proteins']=df['Protein'].copy()
    
    
    # Adds effect of phosphorylation
    pa.addPeptideAndPsitePositions(df, pa.pspFastaFile, pspInput = True)
    pa.addPSPAnnotations(df, pa.pspAnnotationFile)
    pa.addPSPRegulatoryAnnotations(df, pa.pspRegulatoryFile)
    pa.addPSPKinaseSubstrateAnnotations(df, pa.pspKinaseSubstrateFile)
    #pa.addDomains(df, pa.domainMappingFile)
    #pa.addMotifs(df, pa.motifsFile)
    pa.addInVitroKinases(df, pa.inVitroKinaseSubstrateMappingFile)
    

    
    matrix.obs=df.copy()
    return(matrix)

def add_Yasushi_annotations(matrix, Path):
    
    Yas_dict=np.load(f'{Path}/Yasushi_annotations.npy', allow_pickle=True)
    Yas_dict=dict(enumerate(Yas_dict.flatten()))[0]

    matrix.obs['Yasushi']='-'*200
    for i in range(len(matrix)):
        gene=matrix.obs['Gene'][i]
        if gene in Yas_dict.keys():
            matrix.obs['Yasushi'][i]=list_to_str(Yas_dict[gene])
        else:
            matrix.obs['Yasushi'][i]='-'
    return(matrix)

def upstream(df, key, Path):
    Tr=np.load(f'{Path}/Gene_name_translator.npy', allow_pickle=True)
    Tr=dict(enumerate(Tr.flatten()))[0]
    
    if key=='vivo':
        key2='PSP'
    if key=='vitro':
        key2='In Vitro'
    
    
    df[f'Upstream_{key}']=[a for a in df[f'{key2} Kinases']]
    for i in range(len(df)):
        s=df[f'{key2} Kinases'][i]
        ls=s.split(';')

        if len(s)>1:
            df[f'Upstream_{key}'][i]=''

            for j in range(len(ls)):
                st=ls[j]
                if '/' in st:
                    st=st.split('/')[0]
                if '[' in st:
                    st=st.split('[')[0]
                if st in Tr.keys():
                    st=Tr[st]
                df[f'Upstream_{key}'][i]+=st+', '
            df[f'Upstream_{key}'][i]=df[f'Upstream_{key}'][i][:-2]
    return(df)

def add_up_and_downstream_annotations(matrix, Path):
    df=matrix.obs.copy()
    
    
    # Adds downstream annotation

    df['Downstream_activated']='-'*200
    df['Downstream_inhibited']='-'*200
    df['Downstream_unknown']='-'*200

    for i in range(len(matrix)):

        s=df['PSP_ON_PROT_INTERACT'][i]
        df['Downstream_activated'][i]=''
        df['Downstream_inhibited'][i]=''
        df['Downstream_unknown'][i]=''

        if s!='-':            
            ls=df['PSP_ON_PROT_INTERACT'][i].split('; ')

            for j in range(len(ls)):

                if ls[j][-9:]=='(INDUCES)':
                    if df['Downstream_activated'][i]!='':
                        df['Downstream_activated'][i]+=', '
                    df['Downstream_activated'][i]+=ls[j][:-9]
                elif ls[j][-10:]=='(DISRUPTS)':
                    if df['Downstream_inhibited'][i]!='':
                        df['Downstream_inhibited'][i]+=', '
                    df['Downstream_inhibited'][i]+=ls[j][:-10]
                elif ls[j][-14:]=='(NOT_REPORTED)':
                    if df['Downstream_unknown'][i]!='':
                        df['Downstream_unknown'][i]+=', '
                    df['Downstream_unknown'][i]+=ls[j][:-14]
                    
    # Adds upstream annotation
                    
    df=upstream(df, 'vivo', Path)
    df=upstream(df, 'vitro', Path)
    
    matrix.obs=df.copy()
    
    return(matrix)

def apply_basic_filtering(matrix):
    matrix=filter_drugs(matrix).copy()
    matrix=filter_peptides(matrix).copy()
    return(matrix)

def cleanup(matrix):
    df=matrix.obs.copy()
    for i in ['Proteins', 'Start positions', 'End positions', 'Site sequence context', 'Site positions', 'PSP_LT_LIT', 
              'PSP_MS_LIT', 'PSP_MS_CST', 'PSP_ON_FUNCTION', 'PSP_ON_PROCESS', 'PSP_ON_PROT_INTERACT', 
              'PSP_ON_OTHER_INTERACT', 'PSP_NOTES', 'PSP Kinases', 'In Vitro Kinases']:
        del df[i]
        
    for i in ['KEGG_gene', 'KEGG_protein', 'KEGG_combined', 'Yasushi', 'Downstream_activated', 
              'Downstream_inhibited', 'Downstream_unknown', 'Upstream_vivo', 'Upstream_vitro']:
        for j in range(len(matrix)):
            if df[i][j]=='':
                df[i][j]='-'
    matrix.obs=df.copy()
    
    matrix.var.index=np.arange(len(matrix.var))
    matrix.obs.index=np.arange(len(matrix))
    
    matrix.obs.index=[str(a) for a in matrix.obs.index]
    matrix.var.index=[str(a) for a in matrix.var.index]
    
    return(matrix)


def plot_summary(before_filtering, after_filtering):
    import matplotlib.pyplot as plt
    import numpy as np

    #plot 1:
    sns.set(rc={'figure.figsize':(12,5)})
    
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    plt.subplot(1, 2, 1)
    from matplotlib import rcParams


    sns.set_style('darkgrid')
    df=pd.DataFrame({'Number of regulated p-sites':
                     [before_filtering[-1]*before_filtering[2],after_filtering[-1]*after_filtering[2]],
                     'When':['Before filtering', 'After filtering']})

    sns.barplot(data=df, x='When', y='Number of regulated p-sites')

    #plot 2:


    plt.subplot(1, 2, 2)
    ar=before_filtering
    dfb=pd.DataFrame({'Percentage': np.array(ar[:3])*100, 'Classes': ['Unknown', 'Not regulated', 'Regulated'],
                     'When':'Before filtering'})
    ar=after_filtering
    dfa=pd.DataFrame({'Percentage': np.array(ar[:3])*100, 'Classes': ['Unknown', 'Not regulated', 'Regulated'],
                     'When':'After filtering'})
    DF=pd.concat([dfb, dfa])

    sns.barplot(data=DF, x='Classes', y='Percentage', hue='When')

    plt.show()