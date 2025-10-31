import pandas as pd
from sklearn.impute import SimpleImputer
from typing import Tuple
from sklearn.preprocessing import StandardScaler

def load_data(filepath:str) ->pd.DataFrame:
    """
        This function loads the csv file

        Input parameters:
        filepath: need the path of the csv file to do preprocess

        Output:
        returns a dataframe
    """
    try:
        data = pd.read_csv(filepath)
        print("Data Loaded Successfully")
        print(f"Shape of the data {data.shape}")
        return data
    except FileNotFoundError:
        print(f'File Not Found at {filepath}')
        raise

def check_missing_values(data:pd.DataFrame) ->pd.Series:
    """ This function checks for missing values 
        and return the columns which contain missing values

        Input:
        dataframe
        output:
        series i.e column names
    """
    missing = data.isnull().sum()
    if missing.any():
        print("Missing values found")
        print(missing[missing>0])
    else:
        print("No missing values")

def handle_missing_values(data:pd.DataFrame, strategy:str) ->pd.DataFrame:
    """
        This function looks for the missing values
        and fill it woth strategy specified
        stragey can be mean, media or mode

        Input:
        dataframe and strategy

        Output:
        dataframe with no missing values
    """

    #object for SimpleImputer
    imp = SimpleImputer(strategy=strategy)
    #invoke fit_transform using the object for SimpleImputer
    #which will return an array
    imp_array = imp.fit_transform(data)
    #convert the array to dataframe
    cleanData = pd.DataFrame(imp_array, columns= data.columns)
    return cleanData

def scale_features(data:pd.DataFrame) -> Tuple[pd.DataFrame,pd.Series]:
    """ This function scales the features with mean = 0 
        and standard deviation = 1

        Input:
            cleaned dataframe
        output:
            Except target all the other features are 
            standardized
    """
    
    X_features = data.iloc[:,:-1] #all rows, all columns except last
    y = data.iloc[:,-1]
    print(X_features.shape)
    print(y.shape)

    scaler = StandardScaler()
    X_transform = scaler.fit_transform(X_features)
    scaledX=pd.DataFrame(X_transform,columns=X_features.columns)

    return scaledX,y





