from os import listdir
from os.path import isfile, join
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def time_clean(orig):
    if orig['IdPeriod']==1: 
        return 5400 - orig['Time']            
    if orig['IdPeriod']==2: 
        return h2end - orig['Time']

def x_cleanup(orig): 
    return orig['AttackingDirection']*orig.LocationX

def homeground(orig): 
    home = orig['Match'].split(' v ')[0]
    if orig.Player1team == home: 
        return 'yes'
    else: 
        return 'no'

def path15_16():  # Get the path for the data from the 2015-16 EPL season
    """
    Get the path for the data from the 2015-16 EPL season
    Returns: list of address paths of the "events.csv" file (relative to this file)
    """
    mypath = "2015-16/England Premier League/"
    subfolders = [f.path for f in os.scandir(mypath) if f.is_dir()]
    file_dir = []
    idx = []

    for _dir in subfolders:
        subdir = len(next(os.walk(_dir))[1])
        if len(next(os.walk(_dir))[1]) == 0:
            subfolders.remove(_dir)

    for subfolder in subfolders:
        onlyfiles = [f for f in listdir(subfolder) if isfile(join(subfolder, f))]
        for i in range(len(onlyfiles)):
            # print(i)
            if "Event" in onlyfiles[i]:
                file_dir.append((onlyfiles[i], subfolder))
    abs_path = []
    for i in file_dir:
        abs_path.append(i[1] + "/" + i[0])
    return abs_path

def path16_17():  # Get the path for the data from the 2016-17 EPL season
    """
    Get the path for the data from the 2016-17 EPL season
    Returns: list of address paths of the "events.csv" file (relative to this file)
    """
    mypath = "2016-17/England Premier League/"  # only yhe directory here is changed
    subfolders = [f.path for f in os.scandir(mypath) if f.is_dir()]
    file_dir = []

    for subfolder in subfolders:
        onlyfiles = [f for f in listdir(subfolder) if isfile(join(subfolder, f))]
        for i in onlyfiles:
            # print(i)
            if "Event" in i:
                file_dir.append(i)
    abs_path = []
    for i in range(len(subfolders)):
        abs_path.append(
            subfolders[i] + "/" + file_dir[i]
        )  # different logic used here since empty folders

    return abs_path

def combine_all_paths():
    p1 = path15_16()
    p2 = path16_17()
    
    paths = []
    for i in p1:
        paths.append(i)
    for j in p2:
        paths.append(j)
        
    return paths

def create_dataframe(paths):
    col_list = ['ScoreHomeTeam', 'ScoreAwayTeam', 'RedCardsHomeTeam', 'RedCardsAwayTeam', 'timeLeft','AttackingDirection','LocationX', 'LocationY','HomeTeam','PhaseType','isGoal']
    dataframe = pd.DataFrame(columns=col_list)
    for filepath in paths:
        events = pd.read_csv(filepath, header =0)
        # events = events.dropna(subset = ['AttackingDirection'])
        global h2end 
        h2end = events.iloc[-1]['Time']
        events['timeLeft'] = events.apply(time_clean, axis = 1)
        events['x'] = events.apply(x_cleanup, axis = 1)
        events['HomeTeam'] = events.apply(homeground, axis = 1)
        events_subset = events[['EventName','ScoreHomeTeam', 'ScoreAwayTeam', 'RedCardsHomeTeam', 'RedCardsAwayTeam', 'timeLeft','AttackingDirection','LocationX', 'LocationY','PhaseType','HomeTeam',]]
        events_subset = events_subset.dropna()
        try:
            shots_taken = events_subset[events_subset['EventName'].str.contains("Shot")]
            goals_scored = events_subset[events_subset['EventName']=='Goal']
            dataset = pd.concat([shots_taken, goals_scored])
            dataset['isGoal'] = np.where(dataset['EventName']=='Goal', 1, 0)
            dataset = dataset.drop(['EventName'],axis=1)
            dataframe = pd.concat([dataframe, dataset])
        except:
            pass
    # dataframe = dataframe.drop(['Unnamed: 0'],axis=1)
    dataframe.to_csv('Dataset csv files/dataframe.csv')
    return dataframe

def encode_dataframe(dataframe):
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder()

    # OneHotEncoding of categorical predictors (not the response)
    Data_cat = dataframe[['HomeTeam','PhaseType']]
    ohe.fit(Data_cat)
    Data_cat_ohe = pd.DataFrame(ohe.transform(Data_cat).toarray(), 
                                    columns=ohe.get_feature_names(Data_cat.columns))

    # Combining Numeric features with the OHE Categorical features
    Data_num = dataframe[['ScoreHomeTeam', 'ScoreAwayTeam', 'timeLeft', 'AttackingDirection','LocationX', 'LocationY']]
    Data_res = dataframe['isGoal']
    Data_cat_ohe=Data_cat_ohe.set_index(Data_num.index)
    Data_ohe = pd.concat([Data_num, Data_cat_ohe, Data_res], sort = False, axis = 1).reindex(index=Data_num.index)
    Data_ohe.to_csv("Dataset csv files/encoded-dataframe.csv")
    return Data_ohe
  

def split_into_Xy(Data_ohe):
    y = pd.DataFrame(Data_ohe['isGoal'])
    X = pd.DataFrame(Data_ohe.drop('isGoal', axis = 1))
    y=y.astype('int')
    X = X.astype(float)
    # X = np.asarray(X).astype('float32')
    return X,y

def normalize_X(X):
    from sklearn.preprocessing import MinMaxScaler
    data = X.values[:, :-1]
    trans = MinMaxScaler()
    data = trans.fit_transform(data)
    dataset = pd.DataFrame(data)
    dataset.to_csv("Dataset csv files/normalised-X.csv")
    return X

def resample_data(X, y):
    from imblearn.over_sampling import SVMSMOTE
    oversample = SVMSMOTE()
    Xn, yn = oversample.fit_resample(X, y)
    Xn.to_csv("Dataset csv files/oversampled-X.csv")
    yn.to_csv("Dataset csv files/oversampled-y.csv")
    return Xn, yn

    # from imblearn.over_sampling import SMOTE
    # oversample = SMOTE(k_neighbors=2, random_state=42)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # columns = X_train.columns
    # os_data_X, os_data_y = oversample.fit_resample(X_train, y_train)
    # upX = pd.DataFrame(data=os_data_X, columns=columns)
    # upy = pd.DataFrame(data=os_data_y, columns=["isGoal"])
    # return upX, upy
    
