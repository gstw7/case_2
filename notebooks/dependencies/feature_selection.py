import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import chi2_contingency
from boruta import BorutaPy



def remover_colunas_constantes(df):
    const_cols = []
    for i in df.columns:
        if len(df[i].unique()) == 1:
            df.drop(i, axis = 1, inplace= True)
            const_cols.append(i)
    return df, const_cols

def variance_threshold_selector(X, threshold=0.01):
    selector = VarianceThreshold(threshold)
    selector.fit(X)    
    return X[X.columns[selector.get_support(indices=True)]]

def boruta_selector(df, y=None):
    Y = df[y]
    df = df.drop(y,axis=1)
    
    num_feat = df.select_dtypes(include=['int64','float64']).columns.tolist()
    
    pipe_num_tree = Pipeline(steps= [('imputer',SimpleImputer(strategy = 'median'))])
    
    preprocessor_tree = ColumnTransformer( transformers = [('num_preprocessor',pipe_num_tree, num_feat)])
    
    RF  = Pipeline(steps = [('preprocessor_rf', preprocessor_tree),('model_rf',RandomForestClassifier(random_state = 123 ,max_depth =5))])
    
    X = preprocessor_tree.fit_transform(df)
    
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)    
    # Criando o boruta
    feat_selector = BorutaPy(rf, n_estimators='auto',random_state=123, max_iter = 100) # 500 iterações até convergir
    feat_selector.fit(X,Y)
    # Terceiro filtro com as features selecionadas pelo boruta
    cols_drop_boruta= [not x for x in feat_selector.support_.tolist()] # apenas invertendo o vetor de true/false
    cols_drop_boruta= df.loc[:,cols_drop_boruta].columns.tolist()
    return cols_drop_boruta