import pandas as pd
import joblib
from module import *
from preprocess import *
import argparse
import os

def test(path, model, saving_path):
    df = pd.read_csv(path)
    model = joblib.load(model)
    for i in range(6):
        df.insert(0, f'sensor{5+i}', [0]*len(df))
    pred = model.predict(df)

    cols = ['No', 'sensor_point5_i_value', 'sensor_point6_i_value', 'sensor_point7_i_value', 'sensor_point8_i_value', 'sensor_point9_i_value', 'sensor_point10_i_value']
    index = np.array(range(1, 101)).reshape(-1, 1)
    data = pd.DataFrame(np.concatenate([index, pred], axis=1), columns=cols)
    
    if saving_path == None:
        path = '111096_TestResult.csv'
    else:
        path = saving_path + '/' + '111096_TestResult.csv'

    # if os.path.exists(path):
    #     os.remove(path)

    get_logging('Saveing Testing results')
    data.to_csv(path, index=False)

def get_parser():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument(
              '--file_path',
              default='test.csv',
              help='Test file',
              type=str          
                 )

    parser.add_argument(
        '--model',
        default='model.pkl',
        help='Load model',
        type=str
    )
    parser.add_argument(
        '--saving_path',
        default=None,
        help='Path for saving data',
        type=str
        
    )
    return parser.parse_args()





if __name__ =='__main__':
    args = get_parser()
    test(path=args.file_path, model=args.model, saving_path=args.saving_path )
