#%%
import numpy as np
import pandas as pd
import ssl
import random

import warnings
warnings.filterwarnings('ignore')


ssl._create_default_https_context = ssl._create_unverified_context

#%%
def _get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return np.nan

#%%
def get_titanic_dataset():

    url = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"
    data = pd.read_csv(url)
    data = data.replace('?', np.nan)
    data['cabin'] = data['cabin'].apply(_get_first_cabin)

    return data


# To download the Credit Approval dataset from the UCI Machine Learning Repository 
# visit (http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/) 
# and click on crx.data and crx.names to download data and variable names.

#%%
def get_credict_approval_dataset():

    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
    data = pd.read_csv(url,header=None)

    # Crea nombres de variables de acuerdo a la información en UCI Machine Learning Repo 
    varnames = ['A'+str(s) for s in range(1, 17)]

    # Agrega nombres de columnas
    data.columns = varnames

    # Reemplaza ? por np.nan
    data = data.replace('?', np.nan)

    # cambia algunas variables al tipo correcto
    data['A2'] = data['A2'].astype('float')
    data['A14'] = data['A14'].astype('float')

    # Codifica la variable objetivo a binaria
    data['A16'] = data['A16'].map({'+': 1, '-': 0})

    ## Agrega mas valores faltantes en posiciones aleatorias, esto ayudará con los demos en este curso

    #random.seed(9001)

    #values = set([random.randint(0, len(data)) for p in range(0, 100)])

    #mylist=['A3', 'A8', 'A9', 'A10']
    #for var in mylist:
    #    data.loc[values, var] = np.nan
    

    return data
#%%
def get_boston_dataset():
    # Información acerca del conjunto de datos
    # Boston House Prices: Precios de casas en Boston

    # El objectivo es predecir 
    # "el valor mediano de las casas en Boston" 
    # esta es la columna MEDV en los datos

    # el resto de las variables representan características
    # acerca de las casas y de los vencidarios

    ''' Variables en orden:
    CRIM     per capita crime rate by town
    ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    INDUS    proportion of non-retail business acres per town
    CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    NOX      nitric oxides concentration (parts per 10 million)
    RM       average number of rooms per dwelling
    AGE      proportion of owner-occupied units built prior to 1940
    DIS      weighted distances to five Boston employment centres
    RAD      index of accessibility to radial highways
    TAX      full-value property-tax rate per $10,000
    PTRATIO  pupil-teacher ratio by town
    B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    LSTAT    % lower status of the population
    MEDV     Median value of owner-occupied homes in $1000's'''

    #fuente: http://lib.stat.cmu.edu/datasets/boston

     # Carga los datos Boston House price de la fuente
    df = pd.read_csv(
    filepath_or_buffer="http://lib.stat.cmu.edu/datasets/boston",
    delim_whitespace=True,
    skiprows=21,
    header=None,
    )

    #Lista con las Variables Independientes
    columns = [
        'CRIM',
        'ZN',
        'INDUS',
        'CHAS',
        'NOX',
        'RM',
        'AGE',
        'DIS',
        'RAD',
        'TAX',
        'PTRATIO',
        'B',
        'LSTAT',
        'MEDV',
    ]

    # Aplana todos los valores en una única lista larga y elimina los valores nulos
    values_w_nulls = df.values.flatten()
    all_values = values_w_nulls[~np.isnan(values_w_nulls)]

    # Adecúa los valores para que tengan 14 columnas y crea un nuevo dataframe con ellas
    data = pd.DataFrame(
        data = all_values.reshape(-1, len(columns)),
        columns = columns,
    )

    return data

#%% 
def get_pima_dataset():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pd.read_csv(url, names=names)
    return dataframe

#%% Penguins dataset
def get_penguins_dataset():
    preprocessed_penguins_df = pd.read_csv('https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv')
    return preprocessed_penguins_df

#%%
if __name__ == "__main__":
    data=get_credict_approval_dataset() 
    
    print(data.head())
    print("Done")
# %%
