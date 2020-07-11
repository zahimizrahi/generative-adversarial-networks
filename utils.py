from scipy.io import arff
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder


def load_data_from_arff(path):
  """
  loads the given data from arff file located in the given path. returns pandas DataFrame
  """
    data = arff.loadarff('data/adult.arff')
    df = pd.DataFrame(data[0])
    return df


def preprocess_data(data_df, normalized=True, how='standard', class_col='class'):
  """
  transform and normalize the data according to the how argument. gets a data_df and returns a transformed pandas dataframe.
  """
    le = LabelEncoder()
    data_columns = data_df.columns[data_df.columns != class_col].tolist()
    for col in data_columns:
        if data_df[col].dtype != np.float64:
            le.fit(data_df[col])
        data_df[col] = le.transform(data_df[col])
    if normalized:
        if how == 'standard':
            data_df[data_columns] = StandardScaler().fit_transform(data_df[data_columns])
        elif how == 'power':
            data_df[data_columns] = PowerTransformer(method='yeo-johnson', standardize = True, copy=True).fit_transform(data_df[data_columns])
    return data_df