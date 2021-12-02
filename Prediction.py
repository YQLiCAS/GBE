# -*- coding: utf-8 -*-

import scipy
import scipy.stats
from scipy import stats
from random import shuffle
import pandas as pd, numpy as np
from mlxtend.preprocessing import minmax_scaling
import keras
from keras.models import Sequential,model_from_json
from keras.layers.core import Dense, Flatten, Dropout , Activation
from keras.layers.convolutional import Convolution1D, Convolution2D
from tensorflow.keras.optimizers import Adam
from numpy import hstack
from numpy import array
import os, sys
import pickle
import theano
from theano import config as config

def import_data(filepath):
    data = pd.read_csv(filepath)
    return data

def get_X_Y_train_test(filepath,editor,task):
    data = preprocess(filepath,editor,task)
    targets,n20s_l,e_dict = len(data),list(),{'T': 3,'G': 2,'C': 1,'A': 0 }
    split,inds = int(targets*4/5),np.argsort(data['total'])
    for n20 in data['N20']:
        n20_l = list()
        for b in range(4):
            n20_l.append([1 if e_dict[n20[x]] == b else 0 for x in range(20)] )
        n20s_l.append(array(n20_l)[np.newaxis, :])
    X = array(n20s_l).astype(config.floatX)
    train,test = inds[0:split],inds[split:targets]

    try:
        if task == 0:
            Y = yeff(data,editor)
        elif task == 1:
            Y = ypro(data)
    except Exception as e: print(e)
    return X,Y, shuffle(train), test

def preprocess(filepath,editor,task):
    data = import_data(filepath)
    data = data.dropna()
    data = data.reset_index()
    try:
        if task == 0:
            if editor == 0:
                data =  data[data['G/total']!=0]
                data= data.reindex(columns =["N20", "G/total",'total'])
            elif editor == 1:
                data =  data[data['T/total']!=0] 
                data= data.reindex(columns =["N20", "T/total",'total'])
        elif task == 1:
            data = data.reindex(columns =["N20", "A/edit","G/edit","T/edit",'total'])
    except Exception as e: print(e)

    data = data.reset_index()
    if task == 0 and editor == 0:
        target, lmbda = stats.boxcox(data['G/total'])
        data['G/total'] = target
        data['s_G/total'] = minmax_scaling(data['G/total'], columns=[0])
    if task == 0 and editor == 1:
        target, lmbda = stats.boxcox(data['T/total'])
        data['T/total'] = target
        data['s_T/total'] = minmax_scaling(data['T/total'], columns=[0])
    return data

def yeff(data,editor):
    try:
        if editor == 0:
            target = 's_G/total'
        elif editor == 1:
            target = 's_T/total'
    except Exception as e: print(e)
    Y = np.expand_dims(data['s_G/total'], axis=1).astype(config.floatX)
    return Y

def ypro(data):
    y1= [data['A/edit']]
    y3= [data['G/edit']]
    y4= [data['T/edit']]
    y1 = array([y1[0][i] for i in range(len(y1[0]))]).reshape(len(y1[0]),1)
    y3 = array([y3[0][i] for i in range(len(y3[0]))]).reshape(len(y3[0]),1)
    y4 = array([y4[0][i] for i in range(len(y4[0]))]).reshape(len(y4[0]),1)
    Y = hstack((y1,y3,y4))
    return Y

def XGB():
    data = pd.read_csv('./source data/pn_Motif.csv')  
    df = data.rename(columns={"T/total": "target"})
    df =  df[df['target']!=df['target'][0]]
    target, lmbda = stats.boxcox(df['target'])
    df['target'] = target
    df['target'] = minmax_scaling(df['target'], columns=[0])
    x = df.iloc[:,1:]
    y = df['target'] 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0) 
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.06, gamma=1, subsample=0.8, objective='reg:squarederror', booster='gblinear', n_jobs=-1)
    xgb_model.fit(x, y)
    df = pd.DataFrame(xgb_model.coef_)
    return df

def predict_outcome(filepath,editor,task):
    try:
        if task == 0:
            if editor == 0:
                modeljson = './params/ung_efficiency.json'
                modelweight = './params/ung_efficiency.hdf5'     
            elif editor == 1:
                modeljson = './params/ugi_efficiency.json'
                modelweight = './params/ugi_efficiency.hdf5'  
        elif task == 1:
            if editor == 0:
                modeljson = './params/ung_proportion.json'
                modelweight = './params/ung_proportion.hdf5'
            elif editor == 1:
                modeljson = './params/ugi_proportion.json'
                modelweight = './params/ugi_proportion.hdf5'
        model = model_from_json(open(modeljson).read())
        model.load_weights(modelweight)
        model.compile(loss='mean_squared_error', optimizer='adam')
    except Exception as e: print(e)
    
    X,Y, train, test = get_X_Y_train_test(filepath,editor,task)
    Y_pred = model.predict(X,verbose=0) 
    # return pd.DataFrame(Y_pred), 
    
    # return scipy.stats.pearsonr(Y[test].flatten(),Y_pred[test].flatten())[0]
    return Y,Y_pred




