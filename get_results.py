import os
import pandas as pd
import numpy as np

def replace(df):
    indices = df.index.values
    
    for index in indices:
        og = index
        new = index.split('_')[1]
        df.rename(index={og: new}, inplace=True)

def main():
    aucroc_name = "./results/0_AUCROC.csv"
    aucpr_name = "./results/0_AUCPR.csv"
    f1_name = "./results/0_AUCF1.csv"
        
    AUCROC = pd.read_csv(aucroc_name, index_col = 0) 
    AUCPR = pd.read_csv(aucpr_name, index_col = 0)
    F1 = pd.read_csv(f1_name, index_col = 0)
    
    replace(AUCROC)
    replace(AUCPR)
    replace(F1)
    
    for seed in [1,2,3,4]:
        aucroc_name = "./results/" + str(seed) + "_AUCROC.csv"
        aucpr_name = "./results/" + str(seed) + "_AUCPR.csv"
        f1_name = "./results/" + str(seed) + "_AUCF1.csv"
        
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
    AUCROC_rank = AUCROC_mean.rank(1, ascending=False, method='dense', na_option='bottom').astype(int)
    
    AUCPR_mean = AUCPR.groupby(AUCPR.index).mean()
    AUCPR_std = AUCPR.groupby(AUCPR.index).std()
    AUCPR_rank = AUCPR_mean.rank(1, ascending=False, method='dense', na_option='bottom').astype(int)
    
    F1_mean = F1.groupby(F1.index).mean()
    F1_std = F1.groupby(F1.index).std()
    F1_rank = F1_mean.rank(1, ascending=False, method='dense', na_option='bottom').astype(int)
    
    AUCROC_mean.to_csv("./results/AUCROC_mean.csv")
    AUCROC_std.to_csv("./results/AUCROC_std.csv")
    AUCROC_rank.to_csv("./results/AUCROC_rank.csv")
    
    AUCROC_mean.describe().to_csv("./results/AUCROC_mean_description.csv")
    AUCROC_rank.describe().to_csv("./results/AUCROC_rank_description.csv")
    
    AUCPR_mean.to_csv("./results/AUCPR_mean.csv")
    AUCPR_std.to_csv("./results/AUCPR_std.csv")
    AUCPR_rank.to_csv("./results/AUCPR_rank.csv")
    
    AUCPR_mean.describe().to_csv("./results/AUCPR_mean_description.csv")
    AUCPR_rank.describe().to_csv("./results/AUCPR_rank_description.csv")
    
    F1_mean.to_csv("./results/F1_mean.csv")
    F1_std.to_csv("./results/F1_std.csv")
    F1_rank.to_csv("./results/F1_rank.csv")
    
    F1_mean.describe().to_csv("./results/F1_mean_description.csv")
    F1_rank.describe().to_csv("./results/F1_rank_description.csv")


if __name__ == "__main__":
    main()