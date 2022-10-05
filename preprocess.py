from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import re

import os
import csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans 
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import OneClassSVM

import logging

def get_logging(word):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    print('\n')
    logging.info(word)





class Imputer(TransformerMixin):

  def __init__(self):
    super().__init__()


  def fit(self, X, y=None, **fit_params):
    self.cols = X.columns

    return self

  def transform(self, X, y=None):
    X_copy = X.copy()
    imputer1 = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 0 , add_indicator = False)
    imputer2 = SimpleImputer(missing_values = 0, strategy = 'constant', fill_value = 1 , add_indicator = False)
    X_copy = imputer1.fit_transform(X_copy)
    X_copy = imputer2.fit_transform(X_copy)

    return pd.DataFrame(X_copy, columns = self.cols)
    

class AnomalyDetector(TransformerMixin):

  def __init__(self, method = 'svm'):
    super().__init__()
    self.method = method
   
  def fit(self, X, y=None, **fit_params):
    X_copy = X.copy()

    vars = X_copy.iloc[:, 6:]
    targets = X_copy.iloc[:, :6]
    var_cols = vars.columns

    self.scaler = MinMaxScaler()
    vars = pd.DataFrame(self.scaler.fit_transform(vars), columns=var_cols)
    
    if self.method == 'svm':
     self.clf = OneClassSVM()
     self.clf.fit(vars)

    elif self.method == 'lof':
     self.clf = LocalOutlierFactor(novelty=True)
     self.clf.fit(vars)

    elif self.method == 'iforest':
     self.clf = IsolationForest()
     self.clf.fit(vars)

    elif self.method == 'ensemble':
      self.lof = LocalOutlierFactor(novelty=True)
      self.lof.fit(vars)
      self.iforest = IsolationForest()
      self.iforest.fit(vars)
      self.svm = OneClassSVM()
      self.svm.fit(vars)

    return self
  def transform(self, X, y=None):
      
      X_copy = X.copy()
    
      vars = X_copy.iloc[:, 6:]
      targets = X_copy.iloc[:, :6]
      var_cols = vars.columns

      if self.method == 'ensemble':
        pred1 = self.lof.predict(vars)
        pred2 = self.iforest.predict(vars)
        pred3 = self.svm.predict(vars)
        pred = [1 if x==-2 else 0 for x in (pred3+pred2)]
      else:
        vars = pd.DataFrame(self.scaler.transform(vars), columns=var_cols)

        pred = [1 if x==-1 else 0 for x in self.clf.predict(vars)]
      X_copy.insert(X_copy.shape[1], 'anomaly', pred) 

      return X_copy


class Collinearity(TransformerMixin):

  def __init__(self, threshold=0.85):
    super().__init__()
    self.threshold = threshold

  def fit(self, X, y=None, **fit_params):
    X_copy = X.copy()
    vars = X_copy.iloc[:, 6:]
    targets = X_copy.iloc[:, :6]

    process = ['clean', 'oven', 'env', 'painting', 'pca']
    self.drop_list = []
    for word in process:
      word_df = vars[[x for x in vars.columns if re.search(word, x)]]
      corr_matrix = word_df.corr().abs()
      upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
      [self.drop_list.append(column) for column in upper_tri.columns if any(upper_tri[column] > self.threshold)]


    return self

  def transform(self, X, y=None, **fit_params):
    
    X_copy = X.copy()
    df = X_copy.drop(list(set(self.drop_list)), axis=1).reset_index(drop=True)
 
    return df

  def plot_corr_data(self, df, word = None):
    if word is not None:
      match = [x for x in df.columns if re.search(word, x)]
      corr_df = df[match]
    else:
      corr_df = df
    f = plt.figure(figsize=(19, 15))
    plt.matshow(corr_df.corr(), fignum=f.number)
    plt.xticks(range(corr_df.select_dtypes(['number']).shape[1]), corr_df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(corr_df.select_dtypes(['number']).shape[1]), corr_df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)



# class TargetTransformer(TransformerMixin):

#   def __init__(self):
#     super().__init__()

#   def fit(self, X, y=None, **fit_params):
    
#     X_copy = X.copy()
#     vars = X_copy.iloc[:, 6:]
#     targets = X_copy.iloc[:, :6]

#     # LOF
#     clf = LocalOutlierFactor()
#     lof_pred = clf.fit_predict(targets)
#     tf_lof_pred = [False if x==-1 else True for x in lof_pred]


#     # IForest
#     iforest = IsolationForest(n_estimators= 1000,random_state=42)
#     iforest.fit(targets)
#     iforest_pred = iforest.predict(targets)
#     tf_iforest_pred = [False if x==-1 else True for x in iforest_pred]

#     ensemble_pred = lof_pred + iforest_pred
#     tf_ensemble_pred = [False if x==-1 else True for x in ensemble_pred]

#     self.df = X_copy.iloc[tf_lof_pred]
                                   

#     return self

#   def transform(self, X, y=None):
  
#     X_copy = X.copy()
  
#     if X_copy.shape[0] < 200:
#       return X_copy
#     else:
#       return self.df


# class PCATransfomer(TransformerMixin):

#   def __init__(self):
#     super().__init__()
#     self.n_components=15
  
#   def fit(self, X, y=None, **fit_params):
#     X_copy = X.copy()
#     vars = X_copy.iloc[:, 6:]
#     targets = X_copy.iloc[:, :6]


#     self.scaler = MinMaxScaler()
#     scaler_vars = self.scaler.fit_transform(vars.values)
#     self.pca = PCA(n_components=self.n_components)
#     self.pca.fit(scaler_vars)

#     return self

#   def transform(self, X, y=None):
#     X_copy = X.copy()
#     vars = X_copy.iloc[:, 6:]
#     targets = X_copy.iloc[:, :6]

#     X_scaler = self.scaler.transform(vars.values)
#     X_pca = self.pca.transform(X_scaler)

#     for i in range(self.n_components):  # add pca columns in df
#       X_copy[f'pca_added{i}'] = pd.DataFrame(X_pca).iloc[:,i].values


#     return X_copy



# class KmeansCluster(TransformerMixin):

#   def __init__(self):
#     super().__init__()

#   def fit(self, X, y=None, **fit_params):
#     X_copy = X.copy()
#     vars = X_copy.iloc[:, 6:]
#     targets = X_copy.iloc[:, :6]

#     self.kmeans = KMeans(n_clusters=3)
#     self.kmeans.fit(vars)

#     return self

#   def transform(self, X, y=None):
#     X_copy = X.copy()
#     vars = X_copy.iloc[:, 6:]
#     targets = X_copy.iloc[:, :6]

#     col = list(self.kmeans.predict(vars))

#     X_copy.insert(X_copy.shape[1],'cluster', col)
#     X_copy.cluster = X_copy.cluster.apply(lambda x: f'{x}')
#     df = pd.get_dummies(X_copy, columns=['cluster'])

#     return df

# class DuplicatedTransformer(TransformerMixin):
#   ''' not remove '''

#   def __init__(self, strategy=None):
#     super().__init__()
#     self.strategy = strategy

#   def fit(self, X, y=None, **fit_params):
#     X_copy = X.copy().reset_index(drop=True)
#     vars = X_copy.iloc[:, 6:]
#     targets = X_copy.iloc[:, :6]

#     group_cols = list(vars.columns.values)


#     for id, row in X_copy.groupby(group_cols):
#       target = row.values[:, :6]
#       row_len = target.shape[0]

#       if self.strategy==None:
#         #arithmetic mean
#         values = target.sum(axis=0)/(target.shape[0])
#         ind = row.index.values
#         X_copy.iloc[ind, :6] = values
        
#       elif self.strategy=='geo':
#         #Geometric Mean
        
#         for i in range(row_len):
#           value_plus = 1
#           for j in target[:, i]:
#             value_plus = value_plus*j

#           value =  pow(value_plus, 1/len(target[:, i]))
#           values.append(value)
        
#       elif self.strategy=='median':
#         values = []
#         for i in range(row_len):
#           median = statistics.median(target[:, i])
#           values.append(median)
#         ind = row.index.values
#         X_copy.iloc[ind, :6] = values

#       else:
#         rows = []

#         for i in range(row_len):
#           row_ = []
#           for j in range(6):
#             sensor = row.iloc[:, j].values
#             sensor.sort()

#             if row_len > 1: 
#               if (row_len >= 5) and (row_len%2 != 0):
#                 median = int((row_len+1)/2)

#                 value = sensor[[median-2, median-1, median]].sum()/3
                
                
#               elif (row_len >= 5) and (row_len%2 == 0):
#                 median = int(row_len/2)
#                 value = sensor[[median-2, median-1, median, median+1]].sum()/4
              
#               elif row_len == 4:
#                 value = sensor[[1, 2]].sum()/2
              
#               elif row_len == 3:
#                 value = sensor[1]

#               elif row_len == 2:
#                 value = [sensor.sum()/row_len]
#             else:
#               value = sensor[0]

#             # row = sensers' value
#             if isinstance(value, list):
#               row_.append(value[0])
#             else:
#               row_.append(value)
            
#           rows.append(row_)

#         rows = np.transpose(np.array(rows))
     
#         ind = row.index.values
#         X_copy.iloc[ind,:6] = np.transpose(rows)

   
#     self.df = X_copy
#     # print('fit duplicated')

#     return self

#   def transform(self, X, y=None):
#     X_copy = X.copy()

#     if X_copy.shape[0] <200:
#       # print('transform duplicated')
#       return X_copy
#     else:
#       # print('transform duplicated')
#       return self.df











    
