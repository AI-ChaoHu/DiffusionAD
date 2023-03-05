from baseline.PyOD import PYOD
from baseline.DAGMM.run import DAGMM
#from baseline.DevNet.run import DevNet
#from baseline.Supervised import supervised

import argparse
import numpy as np

from diffusion import Diffusion, DiffusionBagging
import os
import pandas as pd
import torch

from myutils import Utils
from ICL import ICL

import sklearn.preprocessing as skp
import sklearn.metrics as skm
from data_generator import DataGenerator
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def low_density_anomalies(test_log_probs, num_anomalies):
    #anomaly_indices = np.argpartition(scores, -num_anomalies)[-num_anomalies:]
    anomaly_indices = np.argpartition(test_log_probs, num_anomalies-1)[:num_anomalies]
    preds = np.zeros(len(test_log_probs))
    preds[anomaly_indices] = 1
    return preds

def main(args):
    seed = args.seed
    
    dir = './results/'
    
    datagenerator = DataGenerator(seed = seed) # data generator
    utils = Utils() # utils function
    
    #dataset_list = ['1_ALOI']
    #dataset_list = ['32_shuttle', '38_thyroid']
    
    dataset_list = os.listdir('datasets/Classical')
    dataset_list.extend(os.listdir('datasets/NLP_by_RoBERTa'))
    dataset_list.extend(os.listdir('datasets/CV_by_ResNet18'))
    #dataset_list.sort()

    model_dict = {}

    # from pyod
    for _ in ['IForest', 'OCSVM', 'CBLOF', 'COPOD', 'ECOD', 'FeatureBagging', 'HBOS', 'KNN', 'LODA',
                      'LOF', 'MCD', 'PCA', 'DeepSVDD']:
        model_dict[_] = PYOD
        
        

    # DAGMM
    #model_dict['DAGMM'] = DAGMM
    
    model_dict = {}
    
    for _ in ['COF', 'SOD', 'SOGAAL', 'MOGAAL', 'MCD']:
        model_dict[_] = PYOD
        
    model_dict['DAGMM'] = DAGMM
    
    model_dict = {}
    
    #model_dict['ICL'] = ICL
    
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
        
        #data['X_train'] = skp.power_transform(data['X_train'])
        #data['X_test'] = skp.power_transform(data['X_test'])
        
        print(data['X_train'][0].size)
        # model initialization
        #clf = Diffusion([data['X_train'][0].size, 512, 1024, 512], binning=True).to(device)
        clf = DiffusionBagging()
        print(clf)
        
        model = clf
        
        # training, for unsupervised models the y label will be discarded
        clf = clf.fit(X_train=data['X_train'], y_train=data['y_train'], X_test = data['X_test'], Y_test = data['y_test'], epochs=420, device=device)
        
        # output predicted anomaly score on testing set
        score = clf.predict_score(data['X_test'], device = device)
        
        indices = np.arange(len(data['y_test']))
        p = low_density_anomalies(-score, len(indices[data['y_test']==1]))
        f1_score = skm.f1_score(data['y_test'], p)

        # evaluation
        result = utils.metric(y_true=data['y_test'], y_score=score)
        
        # save results
        model_name = 'diffusion_420_bins_7_bagging'
        df_AUCROC.loc[dataset, model_name] = result['aucroc']
        df_AUCPR.loc[dataset, model_name] = result['aucpr']
        df_F1.loc[dataset, model_name] = f1_score
        
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
            if name == "ICL":
                name = "ICL_bagging"
            clf = clf(seed=seed, model_name=name)
            
            # training, for unsupervised models the y label will be discarded
            clf = clf.fit(X_train=data['X_train'], y_train=data['y_train'])    
            
            if name == 'DAGMM':
                score = clf.predict_score(data['X_train'], data['X_test'])
            else:
                score = clf.predict_score(data['X_test'])
                
                indices = np.arange(len(data['y_test']))
                p = low_density_anomalies(-score, len(indices[data['y_test']==1]))
                f1_score = skm.f1_score(data['y_test'], p)
                print(f1_score)

                df_F1.loc[dataset, name] = f1_score
                df_F1.to_csv(f1_name)

            inds = np.where(np.isnan(score))
            score[inds] = 0
            
            result = utils.metric(y_true=data['y_test'], y_score=score)
            print(result['aucroc'])
            
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