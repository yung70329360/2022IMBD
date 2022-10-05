from optuna.integration import OptunaSearchCV
import pandas as pd
import joblib
import lightgbm as lgb
import argparse

from module import *
from preprocess import *

def train(path, xgb_path, lgbm_path):
  df = pd.read_csv(path)

  steps = [
      ('SimpleImputer', Imputer()),
      ('AnomalyDetector', AnomalyDetector()),
      # ('TargetTransformer', TargetTransformer()),
    #   ('AvgDuplicated', DuplicatedTransformer(strategy='mix)),
    #   ('KmeansCluster', KmeansCluster()),
    #   ('PCA', PCATransfomer()),
      ('Collinearity', Collinearity(threshold=0.85)),
      ('reg', AutoModel(xgb_path, lgbm_path))
  ]

  pipe = Pipeline(steps=steps, verbose=-1)
  pipe.fit(df)
  save_model(pipe)

def get_parser():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument(
              '--file_path',
              default='train.csv',
              help='training file',
              type=str          
                 )
    parser.add_argument(
        '--xgb_path', 
        default=None,
        help='XGB Model',
        type=str
                )
    parser.add_argument(
        '--lgbm_path',
        default=None,
        help='LGBM Model', 
        type=str
                )
    return parser.parse_args()

def save_model(model):

    get_logging('Saving Model......')
    joblib.dump(model, f'model.pkl')

if __name__ == '__main__':
    args = get_parser()
    train(args.file_path, args.xgb_path, args.lgbm_path)


