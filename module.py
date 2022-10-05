import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.multioutput import MultiOutputRegressor
from optuna.integration import OptunaSearchCV
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
import lightgbm as lgb
import xgboost as xgb
from preprocess import get_logging
import joblib

class AutoModel(TransformerMixin, BaseEstimator):
  def __init__(self, xgb_path, lgbm_path):
    super().__init__()
#     predictor='gpu_predictor' if torch.cuda.is_available() else 'auto'
#     tree_method = 'gpu_hist' if torch.cuda.is_available() else 'auto'
    
    self.xgb_path = xgb_path
    self.lgbm_path = lgbm_path
    
#     self.lgbmodel = lgb.LGBMRegressor(  
#             boosting_type ='dart', 
#             objective = 'rmse',
#             n_estimators = 700, 
#             max_depth=4,
#             min_child_weight=1,
#             num_leaves=8,
#             learning_rate=0.1, 
#             subsample_freq=4,
#             subsample=0.7,
#             colsample_bytree=0.6,
#             reg_alpha=0.00862206896551724,
#             reg_lambda=0.0006989655172413794,
#             random_state=42
#                 )

    self.xgbmodel=  xgb.XGBRegressor(
        booster='dart',
        objective ='reg:squarederror',
#         predictor = 'gpu_predictor', 
#         tree_method = 'gpu_hist', 
        learning_rate=0.1,
        n_estimators=31,
        max_depth=1, 
        min_child_weight=1,
        gamma=0,
        subsample=0.88,
        colsample_bytree=0.74,
        reg_alpha=12,
        reg_lambda=7,
        random_state=42
    )



  def fit(self, X, y=None, **fit_params):
    X_copy = X.copy()
    vars = X_copy.iloc[:, 6:]
    targets = X_copy.iloc[:, :6]
    
    self.xgbReg = MultiOutputRegressor(estimator=self.xgbmodel)
    
    xgbParams = {
        'estimator__n_estimators': IntDistribution(20, 100, log=True),
        'estimator__max_depth': IntDistribution(1, 4),
        'estimator__subsample':FloatDistribution(0.5, 1, log=True),
        'estimator__colsample_bytree':FloatDistribution(0.5, 1, log=True),
        'estimator__reg_alpha':FloatDistribution(5, 15, log=True),
        'estimator__reg_lambda':FloatDistribution(1, 10, log=True),
    }
    

    xgbOpt = OptunaSearchCV(estimator=self.xgbReg, param_distributions = xgbParams, n_trials=1500,timeout=3600*3,n_jobs=-1, cv=10,random_state=42, scoring='neg_root_mean_squared_error', )
    get_logging('Start Training Opt')
    xgbOpt.fit(vars, targets)
    get_logging('Opt finish training')
    
    self.xgbReg.set_params(**xgbOpt.best_params_) 
    get_logging('Start Training Xgb')
    self.xgbReg.fit(vars, targets)
    get_logging('Xgb finish training')
    return self

  def predict(self, X, **predict_params):
  
    X_copy = X.copy()
    vars = X_copy.iloc[:, 6:]
    targets = X_copy.iloc[:, :6]

    xgb_pred = self.xgbReg.predict(vars)
    
    return xgb_pred


