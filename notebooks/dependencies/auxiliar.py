import pandas as pd

def aux(df):

    '''
    in: DataFrame;
    out: DataFrame auxiliar'''
    
    df_aux = pd.DataFrame({'colunas' : df.columns,
                    'tipo': df.dtypes,
                    'missing' : df.isna().sum(),
                    'size' : df.shape[0],
                    'unicos': df.nunique()})
    df_aux['percentual%'] = round(df_aux['missing'] / df_aux['size'],3)*100

    return df_aux

def remover_duplicatas(df):
   
    #Verificando duplicatas nas linhas
    print('Removendo...')
    #Verificando duplicatas colunas
    df_T = df.T
    print(f'Existem {df_T.duplicated().sum()} colunas duplicadas e {df.duplicated().sum()} linhas duplicadas')
    df.drop_duplicates(inplace=True)
    list_duplicated_columns = df_T[df_T.duplicated(keep=False)].index.tolist()
    df_T.drop_duplicates(inplace = True)
    print('Colunas duplicadas:')
    print(list_duplicated_columns)
    
    return  df_T.T, list_duplicated_columns

def converter_tipos(df, dicionario_tipo):
    
    # varrer todas as colunas
    for column in df.columns:
        # converte coluna para tipo indicado no dicionário
        df[column] = df[column].astype(dicionario_tipo[column], errors='ignore')
    
    return df