import pandas as pd
import numpy as np
import seaborn as sns
import anndata
import matplotlib.pyplot as plt
import scipy
from sklearn import metrics, preprocessing
import tensorflow as tf
## for deep learning
from tensorflow.keras import models, layers, utils  #(2.6.0)


### To restrict memory used
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    except RuntimeError as e:
        print(e)

    

def matrix_factorization(matrix, dim0=30):
    n0,n1=matrix.shape

    # Users (1,embedding_size)
    xusers_in = layers.Input(name="xusers_in", shape=(1,))
    xusers_emb = layers.Embedding(name="xusers_emb", input_dim=n0, output_dim=dim0)(xusers_in)
    xusers = layers.Reshape(name='xusers', target_shape=(dim0,))(xusers_emb)
    # Products (1,embedding_size)
    xproducts_in = layers.Input(name="xproducts_in", shape=(1,))
    xproducts_emb = layers.Embedding(name="xproducts_emb", input_dim=n1, output_dim=dim0)(xproducts_in)
    xproducts = layers.Reshape(name='xproducts', target_shape=(dim0,))(xproducts_emb)
    # Product (1)
    xx = layers.Dot(name='xx', normalize=True, axes=1)([xusers, xproducts])
    # Predict ratings (1)
    y_out = layers.Dense(name="y_out", units=1, activation='linear')(xx)
    # Compile
    model = models.Model(inputs=[xusers_in,xproducts_in], outputs=y_out, name="CollaborativeFiltering")
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_percentage_error'])

    return(model)


def matrix_factorization_and_depp_learning(matrix, dim0=30, dim1=30, dim_dl=3):
    n0,n1=matrix.shape


    # Input layer
    xusers_in = layers.Input(name="xusers_in", shape=(1,))
    xproducts_in = layers.Input(name="xproducts_in", shape=(1,))

    # A) Matrix Factorization
    ## embeddings and reshape
    cf_xusers_emb = layers.Embedding(name="cf_xusers_emb", input_dim=n0, output_dim=dim0)(xusers_in)
    cf_xusers = layers.Reshape(name='cf_xusers', target_shape=(dim0,))(cf_xusers_emb)
    ## embeddings and reshape
    cf_xproducts_emb = layers.Embedding(name="cf_xproducts_emb", input_dim=n1, output_dim=dim0)(xproducts_in)
    cf_xproducts = layers.Reshape(name='cf_xproducts', target_shape=(dim0,))(cf_xproducts_emb)
    ## product
    cf_xx = layers.Dot(name='cf_xx', normalize=True, axes=1)([cf_xusers, cf_xproducts])



    # B) Neural Network
    ## embeddings and reshape
    nn_xusers_emb = layers.Embedding(name="nn_xusers_emb", input_dim=n0, output_dim=dim1)(xusers_in)
    nn_xusers = layers.Reshape(name='nn_xusers', target_shape=(dim1,))(nn_xusers_emb)
    ## embeddings and reshape
    nn_xproducts_emb = layers.Embedding(name="nn_xproducts_emb", input_dim=n1, output_dim=dim1)(xproducts_in)
    nn_xproducts = layers.Reshape(name='nn_xproducts', target_shape=(dim1,))(nn_xproducts_emb)
    ## concat and dense
    nn_xx = layers.Concatenate()([nn_xusers, nn_xproducts])
    nn_xx = layers.Dense(name="nn_xx", units=int(dim_dl), activation='relu')(nn_xx)

    # Merge A & B
    y_out = layers.Concatenate()([cf_xx, nn_xx])
    y_out = layers.Dense(name="y_out", units=1, activation='linear')(y_out)
    # Compile
    model = models.Model(inputs=[xusers_in,xproducts_in], outputs=y_out, name="Neural_CollaborativeFiltering")
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_percentage_error'])
    
    return(model)


def mf_and_dl_and_additional_features(matrix, dim0=5, dim1=5, dim_dl=2, dim_features=5, dim_dl_and_features=2):
    n0,n1=matrix.shape


    ################### COLLABORATIVE FILTERING ########################
    # Input layer
    xusers_in = layers.Input(name="xusers_in", shape=(1,))
    xproducts_in = layers.Input(name="xproducts_in", shape=(1,))
    # A) Matrix Factorization
    ## embeddings and reshape
    cf_xusers_emb = layers.Embedding(name="cf_xusers_emb", input_dim=n0, output_dim=dim0)(xusers_in)
    cf_xusers = layers.Reshape(name='cf_xusers', target_shape=(dim0,))(cf_xusers_emb)
    ## embeddings and reshape
    cf_xproducts_emb = layers.Embedding(name="cf_xproducts_emb", input_dim=n1, output_dim=dim0)(xproducts_in)
    cf_xproducts = layers.Reshape(name='cf_xproducts', target_shape=(dim0,))(cf_xproducts_emb)
    ## product
    cf_xx = layers.Dot(name='cf_xx', normalize=True, axes=1)([cf_xusers, cf_xproducts])
    # B) Neural Network
    ## embeddings and reshape
    nn_xusers_emb = layers.Embedding(name="nn_xusers_emb", input_dim=n0, output_dim=dim1)(xusers_in)
    nn_xusers = layers.Reshape(name='nn_xusers', target_shape=(dim1,))(nn_xusers_emb)
    ## embeddings and reshape
    nn_xproducts_emb = layers.Embedding(name="nn_xproducts_emb", input_dim=n1, output_dim=dim1)(xproducts_in)
    nn_xproducts = layers.Reshape(name='nn_xproducts', target_shape=(dim1,))(nn_xproducts_emb)
    ## concat and dense
    nn_xx = layers.Concatenate()([nn_xusers, nn_xproducts])
    nn_xx = layers.Dense(name="nn_xx", units=dim_dl, activation='relu')(nn_xx)


    ######################### CONTENT BASED ############################
    pEC50 = layers.Input(name="pEC50", shape=(1,))
    features0 = layers.Dense(name="features0", units=1, activation='relu')(pEC50)

    SN = layers.Input(name="SN", shape=(1,))
    features1 = layers.Dense(name="features1", units=1, activation='relu')(SN)

    FC = layers.Input(name="FC", shape=(1,))
    features2 = layers.Dense(name="features2", units=1, activation='relu')(FC)

    ######################  Merge additional information     #######################################
    # Merge all
    merge_features = layers.Concatenate()([features0, features1, features2])
    merge_features = layers.Dense(name="merge_features", units=dim_features, activation='relu')(merge_features)


    ######################  Merge additional information     #######################################
    # Merge all
    merger = layers.Concatenate()([nn_xx, merge_features])
    merger = layers.Dense(name="merger", units=dim_dl_and_features, activation='relu')(merger)


    ########################## OUTPUT ##################################
    # Merge all
    
    y_out = layers.Concatenate()([cf_xx, merger])
    y_out = layers.Dense(name="y_out", units=1, activation='linear')(y_out)
    # Compile
    model = models.Model(inputs=[xusers_in, xproducts_in, pEC50, SN, FC], outputs=y_out, name="Hybrid_Model")
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_percentage_error'])
    return(model)




def get_additional_information(matrix, in0, in1):
    
    # Get them all log_transformed
    pEC50=[]
    SN=[]
    FC=[]
    
    for l in range(len(in0)):
        pEC50.append(float(matrix.layers['pEC50'][in0[l],in1[l]]))
        SN.append(np.log(float(matrix.layers['Total SN'][in0[l],in1[l]])))
        FC.append(float(matrix.layers['log(Fold_change_predicted)'][in0[l],in1[l]]))
    
    pEC50=np.array(pEC50)
    SN=np.array(SN)
    FC=np.array(FC)
    
    return(pEC50, SN, FC)


def model_to_output_test_set(matrix, D, model, test_only=False):
    n0,n1=matrix.shape
    C=np.zeros((n0,n1))
    ind0=[]
    ind1=[]
    
    if test_only:
        Dt=D['Test']
        for i in [*Dt['Up-regulated'], *Dt['Down-regulated'],  *Dt['Not regulated']]:
            ind0.append(i[0])
            ind1.append(i[1])

    else:
        for i in range(n0):
            for j in range(n1):
                if matrix.X[i,j]!='Empty':
                    ind0.append(i)
                    ind1.append(j)
                    
    ind0=np.array(ind0)
    ind1=np.array(ind1)
    
    EC50, SN, FC=get_additional_information(matrix, ind0, ind1)
    
    res= model.predict([ind0, ind1, EC50, SN, FC]).T[0]
    
    l=0
    for i in range(len(ind0)):
        C[ind0[i], ind1[i]]= res[i]
    return(C)



def model_to_output(matrix, D, model, test_only=False):
    n0,n1=matrix.shape
    C=np.zeros((n0,n1))
    
    ind0=[]
    ind1=[]
    
    if test_only:
        Dt=D['Test']
        for i in [*Dt['Up-regulated'], *Dt['Down-regulated'],  *Dt['Not regulated']]:
            ind0.append(i[0])
            ind1.append(i[1])

    else:
        for i in range(n0):
            for j in range(n1):
                ind0.append(i)
                ind1.append(j)
                
    ind0=np.array(ind0)
    ind1=np.array(ind1)
    
    res= model.predict([ind0, ind1]).T[0]
    
    for i in range(len(ind0)):
        C[ind0[i],ind1[i]]= res[i]
    return(C)