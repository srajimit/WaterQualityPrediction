from preprocess import load_data, check_missing_values, handle_missing_values,scale_features
from train import training_Model

if __name__ == "__main__":
    water_data = load_data("../data/water_potability.csv")
    #print(water_data.head())
    print("Before cleaning Missing values:")
    check_missing_values(water_data)
    clean_water_data=handle_missing_values(water_data,"mean")
    print("Now check for missing values after cleaning")
    check_missing_values(clean_water_data)
    X,Y=scale_features(clean_water_data)
    #print(X.head())
    #print(Y.head())
    RFM = training_Model(X,Y)
    


