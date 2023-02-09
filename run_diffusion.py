from baseline.PyOD import PYOD
from baseline.DAGMM.run import DAGMM
#from baseline.DevNet.run import DevNet
#from baseline.Supervised import supervised

import argparse
import numpy as np

from diffusion import Diffusion
import os
import pandas as pd
import torch

from myutils import Utils

from data_generator import DataGenerator
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def main(args):
    seed = args.seed
    
    dir = './minmax/'
    
    datagenerator = DataGenerator(seed = seed) # data generator
    utils = Utils() # utils function
    
    #dataset_list = ['1_ALOI']
    #dataset_list = ['32_shuttle', '38_thyroid']
    
    dataset_list = os.listdir('datasets/Classical')
    #dataset_list.sort()

    model_dict = {}

    # from pyod
    for _ in ['IForest', 'OCSVM', 'CBLOF', 'COPOD', 'ECOD', 'FeatureBagging', 'HBOS', 'KNN', 'LODA',
                      'LOF', 'MCD', 'PCA', 'DeepSVDD']:
        model_dict[_] = PYOD

    # DAGMM
    #model_dict['DAGMM'] = DAGMM
    
    model_dict = {}
    
    aucroc_name = dir + str(seed) + "_AUCROC.csv"
    aucpr_name = dir + str(seed) + "_AUCPR.csv"
    f1_name = dir + str(seed) + "_AUCF1.csv"
    
    try:
        df_AUCROC = pd.read_csv(aucroc_name, index_col = 0) 
    except:
        df_AUCROC = pd.DataFrame(data=None, index=dataset_list)
    
    try:
        df_AUCPR = pd.read_csv(aucpr_name, index_col = 0)
    except:
        df_AUCPR = pd.DataFrame(data=None, index=dataset_list)
    try:
        df_F1 = pd.read_csv(f1_name, index_col = 0)
    except:
        df_F1 = pd.DataFrame(data=None, index=dataset_list)
    
    

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for dataset in dataset_list:
        '''
        la: ratio of labeled anomalies, from 0.0 to 1.0
        realistic_synthetic_mode: types of synthetic anomalies, can be local, global, dependency or cluster
        noise_type: inject data noises for testing model robustness, can be duplicated_anomalies, irrelevant_features or label_contamination
        '''
        split_dataset = dataset.split('.')
        if len(split_dataset) == 2:
            dataset = split_dataset[0]
        else:
            dataset = split_dataset[0] + '.' + split_dataset[1]
        print(dataset)
        
        # import the dataset
        datagenerator.dataset = dataset # specify the dataset name
        data = datagenerator.generator(la=0, realistic_synthetic_mode=None, noise_type=None) # only 10% labeled anomalies are available

        print(data['X_train'][0].size)
        # model initialization
        clf = Diffusion([data['X_train'][0].size, 256, 512, 256, 1]).to(device)
        print(clf)
        
        model = clf
        scaler = StandardScaler().fit(data['X_train'])
        data['X_train'] = scaler.transform(data['X_train'])
        data['X_test'] = scaler.transform(data['X_test'])
        
        
        # training, for unsupervised models the y label will be discarded
        clf, f1_score = clf.fit(X_train=data['X_train'], y_train=data['y_train'], X_test = data['X_test'], Y_test = data['y_test'], epochs=500, device=device)
        
        # output predicted anomaly score on testing set
        score = clf.predict_score(data['X_test'], device = device)

        # evaluation
        result = utils.metric(y_true=data['y_test'], y_score=score)
        
        # save results
        df_AUCROC.loc[dataset, 'diffusion'] = result['aucroc']
        df_AUCPR.loc[dataset, 'diffusion'] = result['aucpr']
        df_F1.loc[dataset, 'diffusion'] = f1_score
        
        df_AUCROC.to_csv(aucroc_name)
        df_AUCPR.to_csv(aucpr_name)
        df_F1.to_csv(f1_name)

    for dataset in dataset_list:
        '''
        la: ratio of labeled anomalies, from 0.0 to 1.0
        realistic_synthetic_mode: types of synthetic anomalies, can be local, global, dependency or cluster
        noise_type: inject data noises for testing model robustness, can be duplicated_anomalies, irrelevant_features or label_contamination
        '''
        
        split_dataset = dataset.split('.')
        if len(split_dataset) == 2:
            dataset = split_dataset[0]
        else:
            dataset = split_dataset[0] + '.' + split_dataset[1]
        print(dataset)
        
        # import the dataset
        datagenerator.dataset = dataset # specify the dataset name
        data = datagenerator.generator(la=0.1, realistic_synthetic_mode=None, noise_type=None) # only 10% labeled anomalies are available
        
        for name, clf in model_dict.items():
            # model initialization
            clf = clf(seed=seed, model_name=name)
            
            # training, for unsupervised models the y label will be discarded
            clf = clf.fit(X_train=data['X_train'], y_train=data['y_train'])    
            
            if name == 'DAGMM':
                score = clf.predict_score(data['X_train'], data['X_test'])
            else:
                score = clf.predict_score(data['X_test'])
                f1_score = clf.evaluate(data['X_test'], data['y_test'], device=device)
                df_F1.loc[dataset, name] = f1_score
                df_F1.to_csv(f1_name)

            inds = np.where(np.isnan(score))
            score[inds] = 0
            
            result = utils.metric(y_true=data['y_test'], y_score=score)
            
            # save results
            df_AUCROC.loc[dataset, name] = result['aucroc']
            df_AUCPR.loc[dataset, name] = result['aucpr']
            
            df_AUCROC.to_csv(aucroc_name)
            df_AUCPR.to_csv(aucpr_name)
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--model', type=str,
        default='mlp', help='name of model')
    parser.add_argument('--seed', type=int, 
        default=42, help='random seed')

    args = parser.parse_args()
    main(args)