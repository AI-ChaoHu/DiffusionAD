import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def plot(df, name):
    df.boxplot(figsize=(30, 5))
    plt.savefig('./results/' + name + '.png')


cv_list = ['MNIST-C', 'FashionMNIST', 'CIFAR10', 'SVHN', 'MVTec-AD', '20news', 'agnews']
nlp_list = ['amazon', 'imdb', 'yelp']

drop_list = ['AutoEncoder','VAE', 'diffusion', 'diffusion_no_mean','diffusion_slow','diffusion_4','diffusion_bins','diffusion_l20','diffusion_l2m','diffusion_l24', 'diffusion_l2_quantile','diffusion_400_128','diffusion_420_512','diffusion_bins_mean','diffusion_bins_mean_420','diffusion_bins_mean_420_margin','diffusion_420_huber','diffusion_bins_mean_420_t0', 'diffusion_l2', 'diffusion_l2_64' , 'diffusion_l2_64_200', 'diffusion_t_uniform', 'ICL_noscaling','diffusion_420_bins_8_0.02']

def replace(df):
    indices = df.index.values
    
    for index in indices:
        og = index
        cont = True
        for cv in cv_list:
            if cv in og:
                cont = False
        if not cont:
            continue
        if index not in nlp_list:
            new = index.split('_')[1].lower()
        else:
            new = index.lower()
        df.rename(index={og: new}, inplace=True)

def mean_cv(df):
    df = df.drop(columns = drop_list)
    for dataset in cv_list:
        names = []
        for index in df.index.values:
            if dataset in index:
                names.append(index)
        mean = df.loc[names, :].mean(axis=0)
        df.loc[dataset] = mean
        df = df.drop(index = names)
        
    return df
        
        

def main():
    source = "./results/"
    
    aucroc_name = source + "0_AUCROC.csv"
    aucpr_name = source + "0_AUCPR.csv"
    f1_name = source + "0_AUCF1.csv"
        
    AUCROC = pd.read_csv(aucroc_name, index_col = 0) 
    AUCPR = pd.read_csv(aucpr_name, index_col = 0)
    F1 = pd.read_csv(f1_name, index_col = 0)
    
    replace(AUCROC)
    replace(AUCPR)
    replace(F1)
    
    for seed in [1,2,3]:
        aucroc_name = source + str(seed) + "_AUCROC.csv"
        aucpr_name = source + str(seed) + "_AUCPR.csv"
        f1_name = source + str(seed) + "_AUCF1.csv"
        
        df_AUCROC = pd.read_csv(aucroc_name, index_col = 0) 
        df_AUCPR = pd.read_csv(aucpr_name, index_col = 0)
        df_F1 = pd.read_csv(f1_name, index_col = 0)
        
        replace(df_AUCROC)
        replace(df_AUCPR)
        replace(df_F1)
        
        AUCROC = pd.concat([AUCROC, df_AUCROC])
        AUCPR = pd.concat([AUCPR, df_AUCPR])
        F1 = pd.concat([F1, df_F1])
        
    
    AUCROC_mean = AUCROC.groupby(AUCROC.index).mean()
    AUCROC_std = AUCROC.groupby(AUCROC.index).std()
    
    AUCROC_mean = mean_cv(AUCROC_mean)
    AUCROC_std =  mean_cv(AUCROC_std)
    
    plot(AUCROC_mean, "AUCROC_mean")
    
    print(AUCROC_mean.median(0))
    
    
    AUCROC_rank = AUCROC_mean.rank(1, ascending=False, method='dense', na_option='bottom').astype(int)
    
    AUCPR_mean = AUCPR.groupby(AUCPR.index).mean()
    AUCPR_std = AUCPR.groupby(AUCPR.index).std()
    
    AUCPR_mean = mean_cv(AUCPR_mean)
    AUCPR_std = mean_cv(AUCPR_std)
    
    AUCPR_rank = AUCPR_mean.rank(1, ascending=False, method='dense', na_option='bottom').astype(int)
    
    F1_mean = F1.groupby(F1.index).mean()
    F1_std = F1.groupby(F1.index).std()
    
    F1_mean = mean_cv(F1_mean)
    F1_std = mean_cv(F1_std)
    
    F1_rank = F1_mean.rank(1, ascending=False, method='dense', na_option='bottom').astype(int)
    
    AUCROC_mean.to_csv(source + "AUCROC_mean.csv")
    AUCROC_std.to_csv(source + "AUCROC_std.csv")
    AUCROC_rank.to_csv(source + "AUCROC_rank.csv")
    
    AUCROC_mean.describe().to_csv(source + "AUCROC_mean_description.csv")
    AUCROC_rank.describe().to_csv(source + "AUCROC_rank_description.csv")
    
    AUCPR_mean.to_csv(source + "AUCPR_mean.csv")
    AUCPR_std.to_csv(source + "AUCPR_std.csv")
    AUCPR_rank.to_csv(source + "AUCPR_rank.csv")
    
    AUCPR_mean.describe().to_csv(source + "AUCPR_mean_description.csv")
    AUCPR_rank.describe().to_csv(source + "AUCPR_rank_description.csv")
    
    F1_mean.to_csv(source + "F1_mean.csv")
    F1_std.to_csv(source + "F1_std.csv")
    F1_rank.to_csv(source + "F1_rank.csv")
    
    F1_mean.describe().to_csv(source + "F1_mean_description.csv")
    F1_rank.describe().to_csv(source + "F1_rank_description.csv")


if __name__ == "__main__":
    main()