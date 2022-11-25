import os
import pandas as pd
import numpy as np
import seaborn as sns
import anndata
from scipy import optimize, interpolate
from scipy.stats import f as f_distribution
from scipy.stats import t as t_distribution

doses= np.array([0.0,
   0.03,
   0.3,
   1.0,
   3.0,
   10.0,
   30.0,
   100.0,
   300.0,
   1000.0,
   10000.0])


#### These line shave been copied from Florian Bayers code from Gitlab ######

def logistic_model(x, ec50, slope, front, back):
    """
    logisitc_model(x, ec50, slope, top, bottom)

    Logistic model to fit the drug response data.

    Parameters
    ----------
    x: array-like
        Drug concentrations in log space
    ec50: float
        Inflection point in log space of the sigmoid curve
    slope: float
        slope of the transition between front and back asymptote
    front: float
        front (first) asymptote
    back: float
        back (second) asymptote

    Returns
    -------
    y: array-like
        Drug response
    """
    # predict y with given parameters using the standard model
    return (front - back) / (1 + 10 ** (slope * (x - ec50))) + back


def get_s0(fc_lim, alpha):
    """
    Calculates the s0 value given a log2 fold change limit and an alpha value.
    This is based on the SAM test analysis.
    """
    log_alpha = np.log10(alpha)
    k = fc_lim / (1/2 * log_alpha + 1/10)
    s0 = abs(np.log10(2**k))
    return s0

def map_fc_to_p_samcutoff(x, alpha, s0, deg_f):
    """
    This is the SAM test for a fold change

    x : log2 fold change value

    alpha : significance threshold
    s0 : fudge factor for variance based on SAM-Test idea
    deg_f : degrees of freedom

    Adapted from:
    https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/pmic.201600132
    R Code in the supplement
    """
    # Get the t-value for given significance threshold
    ta = t_distribution.ppf(1 - alpha, deg_f)

    # Mask
    pos = x > (ta * s0)
    neg = x < (-ta * s0)

    # Values
    x_pos = x[pos]
    x_neg = x[neg]
    x_none = x[(~pos) & (~neg)]

    d_pos = x_pos / ta - s0
    d_pos = s0 / d_pos
    d_pos = ta * (1 + d_pos)

    d_neg = x_neg / (-ta) - s0
    d_neg = s0 / d_neg
    d_neg = ta * (1 + d_neg)

    # cumulative density function (cdf) of the Student t distribution
    # Log survival function to have more accurate p value calculations:  - np.log10[(1 - t_dist.cdf(t_val, deg_f)) * 2]
    y_pos = - t_distribution.logsf(d_pos, deg_f) * np.log10(np.e) - np.log10(2)
    y_neg = - t_distribution.logsf(d_neg, deg_f) * np.log10(np.e) - np.log10(2)

    # Combine array
    y_none = np.full(shape=len(x_none), fill_value=np.inf)
    y = pd.concat([pd.Series(y_neg, index=x_neg.index),
                   pd.Series(y_none, index=x_none.index),
                   pd.Series(y_pos, index=x_pos.index)])
    return y[x.index]



def sam_curve(x, deg_f=10-2, fc_lim=np.log2(1.3), alpha=0.015):
    s0=get_s0(fc_lim, alpha)
    ta = t_distribution.ppf(1 - alpha, deg_f)
    d_pos = x / ta - s0
    d_pos = s0 / d_pos
    d_pos = ta * (1 + d_pos)
    y = - t_distribution.logsf(d_pos, deg_f) * np.log10(np.e) - np.log10(2)
    return(y)


def add_fold_change(DF, doses=doses):
    front_predicted=[]
    back_predicted=[]
    fold_change_predicted=[]

    Curve_fits=DF[['Log EC50', 'Curve Slope', 'Curve Front', 'Curve Back']].values

    for i in range(Curve_fits.shape[0]):

        ec50, slope, front, back=Curve_fits[i,:]

        front_predicted.append(front)
        back_predicted.append(logistic_model(np.log10(doses[-1]/10**9), ec50, slope, front, back))
        fold_change_predicted.append(back_predicted[-1]/front)

    DF['Front predicted']=front_predicted
    DF['Back predicted']=back_predicted
    DF['Fold_change_predicted']=fold_change_predicted
    DF['log(Fold_change_predicted)']=np.log(fold_change_predicted)
    DF['0']=0
    
    curve_regualtion_old=['-'*20]*len(DF)
    
    for i in range(len(DF)):
        cr=DF['Curve Regulation'].iloc[i]
        if str(cr)=='nan':
            curve_regualtion_old[i]='Unknown'
        elif str(cr)=='not':
            curve_regualtion_old[i]='Not regulated'
        elif str(cr)=='up':
            curve_regualtion_old[i]='Up-regulated'   
        elif str(cr)=='down':
            curve_regualtion_old[i]='Down-regulated' 
        else:
            curve_regualtion_old[i]=cr
    DF['Old_curve_regulation']=curve_regualtion_old
    return(DF)



def calculate_new_regulation(df, alpha=0.015, fc_lim=0.39, new_not_regualted=True):

    Fold_change_key='Fold_change_predicted'
    Q_value_key='Curve Log P_Value'

    ############## This is a reduced version of Flo's "define_regulated_curves"    ##################################

    fold_change = np.log2(df[Fold_change_key].copy())
    s0 = get_s0(fc_lim, alpha)
    p_value_cutoff = map_fc_to_p_samcutoff(fold_change, alpha=alpha, s0=s0, deg_f=10-2)
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
    df['New_curve_regulation'] = 'Unknown'
    df.loc[not_regulated_mask, 'New_curve_regulation'] = 'Not regulated'
    df.loc[p_mask & up_mask, 'New_curve_regulation'] = 'Up-regulated'
    df.loc[p_mask & down_mask, 'New_curve_regulation'] = 'Down-regulated'
    
    return(df)

def pick_curve(dfsp, pick_criterion='Andromeda Score'):
    
    if pick_criterion=='Percolator Score':
        index_of_max=np.argmax(dfsp['Percolator Score'])
    elif pick_criterion=='Andromeda Score':
        index_of_max=np.argmax(dfsp['Andromeda Score'])
    elif pick_criterion=='Total S/N':
        index_of_max=np.argmax(dfsp['Andromeda Score'])
    elif pick_criterion=='Curve P_Value':
        index_of_max=np.argmin(dfsp['Curve P_Value'])
    else:
        print('Error: pick_criterion is not in [Percolator Score, Andromeda Score, Total S/N, Curve P_value]')
    
    dfss=dfsp.iloc[[index_of_max]]
        
    return(dfss)

def select_curves(df, pick_criterion='Andromeda Score'):
    df['Modified sequence']=df['Modified sequence'].astype('category')

    peptides=[a for a in df['Modified sequence'].cat.categories]
    
    frames=[]

    for peptide in peptides:

        dfp=df[df['Modified sequence']==peptide].copy()

        dfs=pick_curve(dfp, pick_criterion=pick_criterion)

        frames.append(dfs)

    Df=pd.concat(frames)
    Df.index=np.arange(len(Df))
    
    return(Df)


def create_dicts(Data2):
    drugs=[a for a in Data2.keys()]
    
    Pep_dict={}
    Prot_dict={}
    Gene_dict={}
    
    
    peptides=[]

    l=0
    for i in range(len(drugs)):
        drug=drugs[i]
        df=Data2[drug]

        for j in range(len(df)):
            peptide=df['Modified sequence'][j]

            if not peptide in peptides:
                Pep_dict[peptide]=l
                Prot_dict[peptide]=df['Proteins'][j]
                Gene_dict[peptide]=df['Gene Names'][j]
                l+=1
                peptides=[a for a in Pep_dict.keys()]
    return(Pep_dict, Prot_dict, Gene_dict)

def create_regulation_matrix(Data2):
    Pep_dict, Prot_dict, Gene_dict=create_dicts(Data2)
    drugs=[a for a in Data2.keys()]
    n0=len(drugs)
    n1=len(Pep_dict.keys())

    Z=np.full((n1,n0), 'Empty', dtype='<U50')

    for i in range(len(drugs)):

        drug=drugs[i]
        df=Data2[drug]

        for j in range(len(df)):
            Z[Pep_dict[df['Modified sequence'][j]],i]=df['New_curve_regulation'][j]

    matrix=anndata.AnnData(X=Z,  dtype=str)
    matrix.var['Drug']=drugs
    matrix.obs['Modified sequence']=[a for a in Pep_dict.keys()]
    matrix.obs['Proteins']=[Prot_dict[a] for a in Pep_dict.keys()]
    matrix.obs['Gene Names']=[Gene_dict[a] for a in Pep_dict.keys()]

    return(matrix)

def add_key_to_regulation_matrix(Data2, matrix, key):
    
    drugs=[a for a in Data2.keys()]
    
    n0=len(drugs)
    n1=len(matrix)
    
    Pep_dict={}
    for i in range(n1):
        Pep_dict[matrix.obs['Modified sequence'].iloc[i]]=i


    Z=np.full((n1,n0), '-', dtype='<U50')
    
    for i in range(n0):
        drug=drugs[i]
        df=Data2[drug].copy()

        for j in range(len(df)):
            Z[Pep_dict[df['Modified sequence'][j]],i]=df[key][j]

    matrix.layers[key]=Z

    return(matrix)


def recalculate_regulation(Data, fc_lim=0.55, alpha=0.015, pick_criterion='Andromeda Score', new_not_regualted=True):
    Data2={}

    drugs=[a for a in Data.keys()]

    for drug in drugs:

        df=Data[drug].copy()
        df=add_fold_change(df, doses)
        df=calculate_new_regulation(df, alpha=alpha, fc_lim=fc_lim, new_not_regualted=new_not_regualted)
        Df=select_curves(df, pick_criterion=pick_criterion)

        Data2[drug]=Df
    
    return(Data2)


def display_change_in_regualtion_classification(Data2):
    frames=[]
    for drug in Data2.keys():
        df=Data2[drug]

        before=[]
        after=[]
        peptide=[]

        for i in range(len(df)):
            before.append(df['Old_curve_regulation'][i])
            after.append(df['New_curve_regulation'][i])
            peptide.append(df['Modified sequence'][i])

        dfn=pd.DataFrame({'Regulation_prior':before, 'Regulation_posterior':after, 'Drug':drug, 'Peptide':peptide})
        frames.append(dfn)

    Df=pd.concat(frames)
    Df.index=np.arange(len(Df))

    Z=np.zeros((4,4))

    regs=['Unknown','Not regulated','Down-regulated','Up-regulated']
    
    old_reg=Df['Regulation_prior'].astype('category').cat.categories
    new_reg=Df['Regulation_posterior'].astype('category').cat.categories
    
    for i in old_reg:
        if not i in regs:
            print(f'Error: {i} is not in {regs} (old regulation)')
    for i in new_reg:
        if not i in regs:
            print(f'Error: {i} is not in {regs} (new regulation)')


    for i in range(4):
        e0=regs[i]
        for j in range(4):
            e1=regs[j]
            Dfn=Df[Df['Regulation_prior']==e0]
            Z[i,j]=len(Dfn[Dfn['Regulation_posterior']==e1])
    Dfz=pd.DataFrame(Z, columns = ['After: '+a for a in regs])
    Dfz.index=['Before: '+a for a in regs]
    display(Dfz)
    return(Df,Dfz)


def add_curve_fit_results(matrix, Data2):

    for key in ['log(Fold_change_predicted)', 'Total S/N', 'pEC50', 'Log EC50 Error']:
        matrix=add_key_to_regulation_matrix(Data2, matrix, key)

    # When trying to store/load the file with 'Total S/N' as obsm-key this returns an error due to the backslash
    matrix.layers['Total SN']=matrix.layers['Total S/N'].copy()
    del matrix.layers['Total S/N']
    
    return(matrix)


def extract_from_tomls(toml_path, df_path):
    tomls=[a for a in os.listdir(toml_path)]

    Data={}
    for topas in tomls:
        drug=topas[:-5]
        
        A=pd.read_csv(f'{df_path}/{drug}.txt', sep='\t', low_memory=False)
        Ts=A.iloc[np.where([str(a)!='nan' for a in A['Protein Names']])]

        Data[drug]=Ts
    return(Data)


def extract_and_refine_regualtion(toml_path, df_path, fc_lim=0.55, alpha=0.015, pick_criterion='Andromeda Score'):
    
    # Takes about 20 min, and you can safely ignore the DtypeWarning's
    Data=extract_from_tomls(toml_path, df_path)

    # Takes about 90 min to run
    Data2=recalculate_regulation(Data, fc_lim, alpha, pick_criterion, new_not_regualted=False)

    Df,Dfz=display_change_in_regualtion_classification(Data2)

    matrix=create_regulation_matrix(Data2)

    matrix=add_curve_fit_results(matrix, Data2)
    
    return(matrix, Data2, Df, Dfz)