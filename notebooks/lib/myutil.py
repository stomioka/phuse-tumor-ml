# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:23:02 2019

@author: stomioka

updated 2019-10-05
1. addition of impute_df(df, t) as discussed in 2-2-1 to 2-2-5
2. addition of check_missing(xdf,ydf)
3. addition of normalize_df(x_df,y_df)
4. addition of col_median_impute(df, col, m=None)
5. addition of median_impute(df)
6. addition of generate_tr_ts(df1, df2, m, h, seed=2019)
7. addition of load_data()
8. addition of run_models(h_lst, m, grid_param, model )
updated 2019-10-06
1. added plot of mean validation scores of n*cv results to run_models(h_lst, m, grid_param, model)
2. addition of test_accuracy(best_idx, best_models, t, h_lst)
3. addition of compare_test(models, hs, m, t)
4. addition of save_models(df, mo)
5. update impute_df to include additional imputation methods as discussed in 2-2-7

"""
import os
import pickle
import pandas as pd
import numpy as np
from scipy import stats
from impyte import impyte
import warnings

#visuals
from matplotlib import pyplot as plt
import seaborn as sns

#ML models
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split\
, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def corr_matrix(df):
    '''generates correlation matrix
    '''
    corr = df.corr() 
    fig=plt.figure(figsize=(12, 10))
    fig.add_subplot(111)
    plt.title('Tumor Correlation',fontsize=20)
    sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
                cmap='viridis'
                , vmax=1.0
                , vmin=-1.0
                , linewidths=0.1
                , annot=True
                , annot_kws={"size": 8}
                , square=True)


def plot_model_performance(df, t, hue=None):
    '''plot model performance
    @param
    t: is for leading text for title
    '''
    fig, ax = plt.subplots(figsize=(11.7,8.27))
    sns.set_style('white')
    g=sns.boxplot(x='model name', y='accuracy', data=df)
    if hue is None:
        sns.stripplot(x='model name', y='accuracy', data=df, 
                      size=8, jitter=True, edgecolor="gray", linewidth=2)
    else:
        sns.stripplot(x='model name', y='accuracy', data=df, hue=hue,
                      size=8, jitter=True, edgecolor="gray", linewidth=2)
    
    fig.suptitle(t+' accuracy of each model', fontsize=25)
    plt.xticks(rotation=45, fontsize=14)
    g.set_xlabel("Model Name",fontsize=14)
    g.set_ylabel("accuracy",fontsize=14)
    g.tick_params(labelsize=14)
    plt.show()

def check_missing(xdf,ydf):
    '''
    return count of missing for each column
    @param
    xdf: input x pandas df
    ydf: input y pandas df
    @test
    check_missing(central_i,site_i)
    '''
    for c in xdf.columns:
        print ('Number of missing in {}: {}'\
               .format(c, len(xdf[xdf[c].isnull()])))
    print('************************')
    for c in ydf.columns:

        print ('Number of missing in {}: {}'\
               .format(c, len(ydf[ydf[c].isnull()])))
        
def impute_df(df, h=None, method=None):
    '''
    @param
    h: hyperparemter to adjust the model, float
    
    @test
    central_i = impute_df(central, -100)
    site_i    = impute_df(site, -100)
    print('Central: {}\nSite: {}'.format(central_i.shape, site_i.shape))

    '''
    if method is not None:
        h=None
        if method=='median':
            df1=median_impute(df)
            
        if method=='sgd':
            imp = impyte.Impyter(df)
            df1=imp.impute(estimator='sgd', multi_nans=True , threshold={"r2": .85, "f1_macro": .85})
        if method=='knn':
            imp = impyte.Impyter(df)
            df1=imp.impute(estimator='knn', multi_nans=True , threshold={"r2": .85, "f1_macro": .85})
        if method=='bayes':
            imp = impyte.Impyter(df)
            df1=imp.impute(estimator='bayes', multi_nans=True , threshold={"r2": .85, "f1_macro": .85})
        if method=='dt':
            imp = impyte.Impyter(df)
            df1=imp.impute(estimator='dt', multi_nans=True , threshold={"r2": .85, "f1_macro": .85}) 
        if method=='gb':
            imp = impyte.Impyter(df)
            df1=imp.impute(estimator='dt', multi_nans=True , threshold={"r2": .85, "f1_macro": .85}) 
    if method is None:

        if h is not None:
            #impute NADIR
            df1=df.copy()
            df1['NADIR']=df.apply(lambda x: x['BSUM'] if pd.isnull(x['NADIR']) else x['NADIR'], axis=1)
            #impute SUMDIAM
            df1['SUMDIAM']=df.apply(lambda x: h if pd.isnull(x['SUMDIAM']) else x['SUMDIAM'], axis=1)
            #impute PCBSD
            df1['PCBSD']=df1.apply(lambda x: 100*(x['SUMDIAM']-x['BSUM'])/x['BSUM'], axis=1)
            #impute NEWLSN
            df1['NEWLSN']=df.apply(lambda x: 0 if pd.isnull(x['NEWLSN']) else x['NEWLSN'], axis=1)
        if h is None:
            print('h should be deinfed when imputation method is not defined')

    return df1



def col_median_impute(df, col, m=None):
    if m is None:
        m=df[col].median()
    df1=df.copy()
    df1[col]=df.apply(lambda x: m if pd.isnull(x[col]) else x[col], axis=1)
    return df1, m

def median_impute(df):
    '''impute median value
    @param
    df: input df
    
    @test
    central_im = median_impute(central)
    site_im = median_impute(site)
    '''
    df1=df.copy()
    for col in df.columns[df.isna().any()].tolist():
        df1, _ = col_median_impute(df1,col)
    return df1


def normalize_df(x_df,y_df):
    '''normalize'''


    x = np.array(x_df)
    y = np.array(y_df).ravel()
    x = normalize(x,norm='l2', axis=1)  
    return x, y

def generate_tr_ts(df1, df2, m, h=None, method=None, seed=2019):
    '''
    generates training and test data from central and site
    
    @param
    m: method
    h: static value of SUMDIAM to replace missing, adjust it to find the best model

     ----------------------------------------------------------
     2.1.1 - Model based on all of central assessments. 
     Test on site assessments.
     ----------------------------------------------------------
    '''

    if m==1:
        if method is not None:
            h=None
            tr_x = impute_df(df1, method=method).iloc[:,0:7]
            tr_y = df1.iloc[:,7:8]
            ts_x = impute_df(df2, method=method).iloc[:,0:7]
            ts_y = df2.iloc[:,7:8]
            ts_x2, ts_y2 =np.zeros(shape=(1,7)), np.zeros(shape=(1,1))
            
        if h is not None:
            tr_x = impute_df(df1, h).iloc[:,0:7]
            tr_y = df1.iloc[:,7:8]
            ts_x = impute_df(df2, h).iloc[:,0:7]
            ts_y = df2.iloc[:,7:8]
            ts_x2, ts_y2 =np.zeros(shape=(1,7)), np.zeros(shape=(1,1))
      
    ''' ----------------------------------------------------------        
     2.1.1 - Model based on all of central assessments. 
     Test on site assessments.
     ----------------------------------------------------------'''   
    
    if m==2:
        if method is not None:
            h=None
            tr_x= impute_df(df2, method=method).iloc[:,0:7]
            tr_y= df2.iloc[:,7:8]
            ts_x= impute_df(df1, method=method).iloc[:,0:7]
            ts_y= df1.iloc[:,7:8]
            ts_x2, ts_y2 = np.zeros(shape=(1,7)), np.zeros(shape=(1,1))             
            
        if h is not None:
            tr_x= impute_df(df2, h).iloc[:,0:7]
            tr_y= df2.iloc[:,7:8]
            ts_x= impute_df(df1, h).iloc[:,0:7]
            ts_y= df1.iloc[:,7:8]
            ts_x2, ts_y2 = np.zeros(shape=(1,7)), np.zeros(shape=(1,1))    
    
    '''----------------------------------------------------------       
    2.1.3 - Model based on central+site and use 85% of data from each. 
    Test on remaining central assessments, 
    Test on remaining site assessments independently.
    ----------------------------------------------------------'''      
    if m==3:
        if method is not None:
            h=None        
            idx = np.random.RandomState(seed).permutation(len(df1))

            training_idx, test_idx = idx[:round(.85*len(np.array(df1)))], idx[round(.85*len(np.array(df1))):]
            tr_x_3a, ts_x = impute_df(df1, method=method).iloc[training_idx,0:7], impute_df(df1, method=method).iloc[test_idx,0:7]
            tr_y_3a, ts_y = df1.iloc[training_idx,7:8], df1.iloc[test_idx,7:8]

            idx = np.random.RandomState(seed).permutation(len(df2))

            training_idx, test_idx = idx[:round(.85*len(np.array(df2)))], idx[round(.85*len(np.array(df2))):]
            tr_x_3b, ts_x2 = impute_df(df2, method=method).iloc[training_idx,0:7], impute_df(df2, method=method).iloc[test_idx,0:7]
            tr_y_3b, ts_y2 = df2.iloc[training_idx,7:8], df2.iloc[test_idx,7:8]

            tr_x=tr_x_3a.append(tr_x_3b, ignore_index=True,sort=False)
            tr_y=tr_y_3a.append(tr_y_3b, ignore_index=True,sort=False)        
        
        
        if h is not None:
            idx = np.random.RandomState(seed).permutation(len(df1))

            training_idx, test_idx = idx[:round(.85*len(np.array(df1)))], idx[round(.85*len(np.array(df1))):]
            tr_x_3a, ts_x = impute_df(df1, h).iloc[training_idx,0:7], impute_df(df1, h).iloc[test_idx,0:7]
            tr_y_3a, ts_y = df1.iloc[training_idx,7:8], df1.iloc[test_idx,7:8]

            idx = np.random.RandomState(seed).permutation(len(df2))

            training_idx, test_idx = idx[:round(.85*len(np.array(df2)))], idx[round(.85*len(np.array(df2))):]
            tr_x_3b, ts_x2 = impute_df(df2, h).iloc[training_idx,0:7], impute_df(df2, h).iloc[test_idx,0:7]
            tr_y_3b, ts_y2 = df2.iloc[training_idx,7:8], df2.iloc[test_idx,7:8]

            tr_x=tr_x_3a.append(tr_x_3b, ignore_index=True,sort=False)
            tr_y=tr_y_3a.append(tr_y_3b, ignore_index=True,sort=False)

    # normalization
    tr_x, tr_y = normalize_df(tr_x, tr_y)
    ts_x, ts_y = normalize_df(ts_x, ts_y)

    if len(ts_x2)>1:
        
        ts_x2, ts_y2 = normalize_df( ts_x2, ts_y2)

    #print('Training set:\n', len(tr_x),'\nTwo test sets:\n', len(ts_x), '\n', len(ts_x2), )        
    return tr_x, tr_y, ts_x, ts_y, ts_x2, ts_y2

def load_data():
    '''load data'''
    central=pd.read_excel(os.path.join('..','data','tumor0central0imp.xls'))
    site=pd.read_excel(os.path.join('..','data','tumor0site0imp.xls'))
    col_orders = ['BSUM','SUMDIAM','PCBSD','NADIR','ACNSD','PCNSD','NEWLSN','TRGRESP','EVAL']
    central=central[col_orders]
    site   =site[col_orders]
    return central, site

def run_models(m, grid_param, model, method=None, h_lst=None ):
    '''
    train model, validate model, plot validation scores from the best model
    
    @param
    h: imputation value for SUMDIAM, adjust if to get the best model
    m: method 1, 2, 3
        1. Model based on all of central assessments. Test on site assessments.
        2. Model based on all of site assessments. Test on central assessments.
        3. Model based on central+site with 85% of data from each. Test on remaining
           central assessments, Test on remaining site assessments independently.
    grid_param: grid parameters to pass to the model
    model: 'rf', 'svc', 'lr', 'knn','xgb'
    method: imputation method 'median', 'sgd', 'knn','bayes','dt','gb'
    
    '''
    central, site=load_data()

    print('----Training Method: {}----'.format(m))
    best_models=[] #store models
    test_x_set_lst=[] #stores test sets based on different h
    best_scores=[] #stores validation scores
    best_parms=[]  #stores parameters from the best models
    i=0
    if h_lst is None:
        tr_x1, tr_y1, _,_, _,_ = generate_tr_ts(df1=central, df2=site, m=m, method=method, seed=2019)
        
        if model=='rf':    
            clf = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=2019)
                                 , param_distributions=grid_param
                                 , scoring='accuracy'
                                 , n_iter=20
                                 , cv=3
                                 , n_jobs=-1
                                 , refit=True
                                 , iid=True
                                 , return_train_score=True)
        if model=='svc': 
            clf = RandomizedSearchCV(estimator=SVC()
                  , param_distributions=grid_param
                  , scoring='accuracy'
                  , cv=3
                  , n_iter=30
                  , n_jobs=-1
                  , refit=True
                  , iid=True
                  , return_train_score=True)

        if model=='lr': 
            clf = RandomizedSearchCV(estimator=LogisticRegression(penalty='l2',
                                                           multi_class='multinomial')
                              , param_distributions=grid_param
                              , scoring='accuracy'
                              , cv=3
                              , n_iter=30
                              , n_jobs=-1
                              , refit=True
                              , iid=True
                              , return_train_score=True)
        if model=='knn':
            clf = GridSearchCV(estimator=KNeighborsClassifier(n_neighbors = 5
                                                             ,algorithm ='auto')
                              , param_grid=grid_param
                              , scoring='accuracy'
                              , cv=3
                              , refit=True
                              , n_jobs=-1
                              , iid=True
                              , return_train_score=True)

        if model=='xgb':
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=2019)
            clf = RandomizedSearchCV(XGBClassifier(objective='multi:softprob'
                                                      , silent=True
                                                      , nthread=-1
                                                      , n_estimators=2000
                                                      , learning_rate=0.01
                                                      , min_child_weight=1

                                                      )
                                                  , param_distributions=grid_param
                                                  , n_iter=20
                                                  , scoring='accuracy'
                                                  , n_jobs=-1
                                                  , cv=skf.split(tr_x1, tr_y1)
                                                  , verbose=3
                                                  , random_state=2019
                                                  , refit=True
                                                  , iid=True
                                                  , return_train_score=True)

        clf.fit(tr_x1, tr_y1)
        best_models.append(clf)
        best_scores.append(clf.best_score_)
        best_parms.append(clf.best_params_)
        model_name=type(clf.best_estimator_).__name__        
        print('----------------------\n{} {}  \n        imputation: {}\n           Parameters: {}\n       Validation Acc: {}'.format(model_name,i,method , clf.best_params_, clf.best_score_))
        i+=1
        idx=0
    
    if h_lst is not None:
        for h in h_lst:
            #print('h: {}'.format(h))
            tr_x1, tr_y1, _,_, _,_ = generate_tr_ts(df1=central, df2=site, m=m, h=h, seed=2019)
            if model=='rf':    
                clf = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=2019)
                                     , param_distributions=grid_param
                                     , scoring='accuracy'
                                     , n_iter=20
                                     , cv=3
                                     , n_jobs=-1
                                     , refit=True
                                     , iid=True
                                     , return_train_score=True)
            if model=='svc': 
                clf = RandomizedSearchCV(estimator=SVC()
                      , param_distributions=grid_param
                      , scoring='accuracy'
                      , cv=3
                      , n_iter=30
                      , n_jobs=-1
                      , refit=True
                      , iid=True
                      , return_train_score=True)

            if model=='lr': 
                clf = RandomizedSearchCV(estimator=LogisticRegression(penalty='l2',
                                                               multi_class='multinomial')
                                  , param_distributions=grid_param
                                  , scoring='accuracy'
                                  , cv=3
                                  , n_iter=30
                                  , n_jobs=-1
                                  , refit=True
                                  , iid=True
                                  , return_train_score=True)
            if model=='knn':
                clf = GridSearchCV(estimator=KNeighborsClassifier(n_neighbors = 4
                                                                 ,algorithm ='auto')
                                  , param_grid=grid_param
                                  , scoring='accuracy'
                                  , cv=3
                                  , refit=True
                                  , n_jobs=-1
                                  , iid=True
                                  , return_train_score=True)

            if model=='xgb':
                skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2019)
                clf = RandomizedSearchCV(XGBClassifier(objective='multi:softprob'
                                                          , silent=True
                                                          , nthread=-1
                                                          , n_estimators=2000
                                                          , learning_rate=0.01
                                                          , min_child_weight=1

                                                          )
                                                      , param_distributions=grid_param
                                                      , n_iter=20
                                                      , scoring='accuracy'
                                                      , n_jobs=-1
                                                      , cv=skf.split(tr_x1, tr_y1)
                                                      , verbose=3
                                                      , random_state=2019
                                                      , refit=True
                                                      , iid=True
                                                      , return_train_score=True)

            clf.fit(tr_x1, tr_y1)
            best_models.append(clf)
            best_scores.append(clf.best_score_)
            best_parms.append(clf.best_params_)
            model_name=type(clf.best_estimator_).__name__        
            print('----------------------\n{} {}  \n              h value: {}\n           Parameters: {}\n       Validation Acc: {}'.format(model_name,i,h , clf.best_params_, clf.best_score_))
            i+=1
        # obtain best val acc   

        idx=best_scores.index(np.array(best_scores).max())
        if h_lst is not None:
            h_     =h_lst[idx]
        if h_lst is None:
            h_     =method

        parms_ =best_parms[idx]
        acc_=best_scores[idx]
        print('----------------------\nBest {} Model: {}\n        Parameters: {} \n                 h: {} \n    Validation Acc: {}'.format(model_name, idx, parms_,h_, acc_))

    val_scores=best_models[idx].cv_results_['mean_test_score']
    train_scores=best_models[idx].cv_results_['mean_train_score']
    df=pd.DataFrame(
        {'iterations': np.arange(0,len(val_scores),1),
         'train acc': train_scores,
         'validation acc': val_scores,
        })
    df=df.melt(id_vars='iterations',value_name='acc',var_name='train/val')
    sns.lineplot(x='iterations', y='acc', hue='train/val', data=df)
    
    return best_scores, best_parms, best_models, idx

def test_accuracy(best_idx, best_models, t, h_lst):
    '''plot test accuracy of the best ML algorithm from each of 4 test sets.
    'm=1, test=A','m=2, test=B', 'm=3, test=C', 'm=3, test=D'
    
    @param
    best_idx: list of index from best models, Expect the length of 4
    best_models: list of best models. Expect the length of 4
    t:
    h_lst:  a list of imputation values for SUMDIAM, float
    '''
    central, site=load_data()
    test_acc=[]
    h_=[] 
    for m, idx in enumerate(best_idx):
        mo = best_models[m]
        m+=1
        h_.append(h_lst[idx])
        tr_x, tr_y, test_x,test_y, _,_ = generate_tr_ts(df1=central, df2=site, m=m, h=h_lst[idx], seed=2019)
        clf=mo[idx].best_estimator_
        pred=clf.predict(test_x)
        test_acc.append(accuracy_score(test_y, pred))
        if m==3:
            h_.append(h_lst[idx])
            tr_x, tr_y, _,_, test_x,test_y = generate_tr_ts(df1=central, df2=site, m=m, h=h_lst[idx], seed=2019)
            clf=mo[idx].best_estimator_
            pred=clf.predict(test_x)
            test_acc.append(accuracy_score(test_y, pred))
    df=pd.DataFrame(
    {'accuracy': test_acc,
     'model name': ['m=1, test=A','m=2, test=B', 'm=3, test=C', 'm=3, test=D'],
     'h': h_
    })
    plot_model_performance(df, t, hue='h')
    return test_acc

def compare_test(models, hs, m, t):
    central, site=load_data()
    entries = []
    pred_lst =[]
    for idx, model in enumerate(models):
        h=hs[idx]
        model_name = type(model.best_estimator_).__name__ 

        _,_,X_ts1, y_ts1, _,_ =generate_tr_ts(df1=central, df2=site, m=m, h=h, seed=2019)

        y_pred=model.predict(X_ts1)
        acc=accuracy_score(y_ts1, y_pred)
        if m==1:
            test_set='A'
        if m==2:
            test_set='B'
        if m==3:
            test_set='C'
            
        entries.append((model_name, h, test_set, acc))
        pred_lst.append(y_pred)
        if m ==3: 
            _,_,_,_,X_ts2, y_ts2 =generate_tr_ts(df1=central, df2=site, m=3, h=h, seed=2019)

            y_pred=model.predict(X_ts2)
            acc=accuracy_score(y_ts2, y_pred)
            test_set='D'
            entries.append((model_name, h, test_set,acc))
            pred_lst.append(y_pred)

    test_df = pd.DataFrame(entries, columns=['model name', 'h', 'test_set', 'accuracy'])
    plot_model_performance(test_df, t, hue='test_set')
    return test_df

def save_models(df, mo):
    '''save models
    @param
    df: data frame with following columns ['model name', 'accuracy']
    mo: list of models. Output of _, _, best_models, _=run_models()
        e.g. mo=[rf1_best_models, rf2_best_models, rf3_best_models]
    '''
    for cv, model_name in zip(mo, list(set(df['model name']))):
        results=np.array(df[df['model name']==model_name]['accuracy'])
        i=list(results).index(results.max())
        clf=cv[i].best_estimator_
        pickle.dump(clf, open(os.path.join('../models', model_name+'-central-site.sav'), 'wb'))

        #to load
        #clf = pickle.load(open('../models/rf_site-central-model.sav', 'rb'))
        
def plot_hist(history):
    '''plot history acc and loss
    @param
    history: history from tf
    '''
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy (batch size: {})'.format(history.params['batch_size']))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss (batch size: {})'.format(history.params['batch_size']))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()