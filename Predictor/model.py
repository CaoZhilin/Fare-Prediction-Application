#!/usr/bin/env python
# coding: utf-8

# Imports
import pandas as pd
import math
import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge,BayesianRidge  
from sklearn.cluster import MiniBatchKMeans  


def haversine_distance(origin, destination):
    """
    # Calculate the spherical distance between 2 coordinates, with each specified as a (lat, lng) tuple

    :param origin: (lat, lng)
    :type origin: tuple
    :param destination: (lat, lng)
    :type destination: tuple
    :return: haversine distance
    :rtype: float
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d

def direction(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km
    
    dlon = math.radians(lon2 - lon1)
    
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = math.sin(dlon) * math.cos(lat2)
    b = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return np.degrees(np.arctan2(a, b))



def process_train_data(raw_df):
    """
    :param raw_df: the DataFrame of the raw training data
    :return:  a DataFrame with the predictors created
    """
    # drop null
    raw_df.dropna(inplace=True)
    # get distance
    raw_df['distance'] = raw_df.apply(lambda row: haversine_distance((row.pickup_latitude, row.pickup_longitude), (row.dropoff_latitude, row.dropoff_longitude)), axis=1)
    # drop abnormal data
    df = raw_train[(raw_train.distance > 0) 
                   & (raw_train.fare_amount >= 2.5) 
                   & (raw_train.passenger_count.between(0,6)) 
                   & (raw_train.pickup_longitude.between(-80, -70))
                   & (raw_train.dropoff_longitude.between(-80, -70)) 
                   & (raw_train.pickup_latitude.between(35, 45)) 
                   & (raw_train.dropoff_latitude.between(35, 45))].copy()
    # get fare per distence by fare_amount / distance
    df['fare_dist'] = df.apply(lambda row: (row.fare_amount)/row.distance, axis=1)
    # drop abnormal fare
    df = df[df.fare_dist < 100].copy()

    df['direction'] = df.apply(lambda row: direction((row.pickup_latitude, row.pickup_longitude), (row.dropoff_latitude, row.dropoff_longitude)), axis=1)
    
    df['year'] = df.apply(lambda row: row.pickup_datetime.year, axis=1)
    df['hour'] = df.apply(lambda row: row.pickup_datetime.hour, axis=1)
    df['day'] = df.apply(lambda row: row.pickup_datetime.weekday(), axis=1)

    
    df['drop_jfk_dist'] = df.apply(lambda row: jfk_dist((row.dropoff_latitude, row.dropoff_longitude)), axis=1)
    df['pick_jfk_dist'] = df.apply(lambda row: jfk_dist((row.pickup_latitude, row.pickup_longitude)), axis=1)

#     df['drop_lga_dist'] = df.apply(lambda row: lga_dist((row.dropoff_latitude, row.dropoff_longitude)), axis=1)
#     df['pick_lga_dist'] = df.apply(lambda row: lga_dist((row.pickup_latitude, row.pickup_longitude)), axis=1)

#     df['drop_new_dist'] = df.apply(lambda row: new_dist((row.dropoff_latitude, row.dropoff_longitude)), axis=1)
#     df['pick_new_dist'] = df.apply(lambda row: new_dist((row.pickup_latitude, row.pickup_longitude)), axis=1)
    
#     df['drop_man_dist'] = df.apply(lambda row: man_dist((row.dropoff_latitude, row.dropoff_longitude)), axis=1)
#     df['pick_man_dist'] = df.apply(lambda row: man_dist((row.pickup_latitude, row.pickup_longitude)), axis=1)
    
    
    df['manhat_dist'] = df.apply(lambda row: manhat_dist((row.pickup_latitude, row.pickup_longitude), (row.dropoff_latitude, row.dropoff_longitude)), axis=1)
    df['abs_lat'] = df.apply(lambda row: abs(row.pickup_latitude - row.dropoff_latitude), axis=1)
    df['abs_lon'] = df.apply(lambda row: abs(row.pickup_longitude - row.dropoff_longitude), axis=1)
    df['euclidean'] = df.apply(lambda row: math.sqrt(row.abs_lat**2 + row.abs_lon**2), axis=1)

    df = df.drop(['fare_dist'], axis=1)

    return df

def jfk_dist(coordinate):
    jfk_dist = haversine_distance((40.6413, -73.7781), coordinate)
    return jfk_dist

def lga_dist(coordinate):
    lag_dist = haversine_distance((40.7769, -73.8740), coordinate)
    return lag_dist

def new_dist(coordinate):
    new_dist = haversine_distance((40.6895, -74.1745), coordinate)
    return new_dist

def man_dist(coordinate):
    man_dist = haversine_distance((40.7589, -73.9851), coordinate)
    return man_dist


def euclid_dist(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    abs_lat = abs(lat1 - lat2)
    abs_lon = abs(lon1 - lon2)
    return math.sqrt(abs_lat**2 + abs_lon**2)
    
def manhat_dist(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    lon_dist = haversine_distance((lat1, lon1), (lat2, lon1))
    lat_dist = haversine_distance((lat2, lon1), (lat2, lon2))
    return lon_dist + lat_dist

def process_test_data(raw_df):
    """
    :param raw_df: the DataFrame of the raw test data
    :return: a DataFrame with the predictors created
    """
    raw_df['distance'] = raw_df.apply(lambda row: haversine_distance((row.pickup_latitude, row.pickup_longitude), (row.dropoff_latitude, row.dropoff_longitude)), axis=1)
#     raw_df['fare_dist'] = raw_df.apply(lambda row: row.fare_amount/row.distance, axis=1)
    raw_df['direction'] = raw_df.apply(lambda row: direction((row.pickup_latitude, row.pickup_longitude), (row.dropoff_latitude, row.dropoff_longitude)), axis=1)
    raw_df['year'] = raw_df.apply(lambda row: row.pickup_datetime.year, axis=1)
    raw_df['hour'] = raw_df.apply(lambda row: row.pickup_datetime.hour, axis=1)
    raw_df['day'] = raw_df.apply(lambda row: row.pickup_datetime.weekday(), axis=1)

    raw_df['drop_jfk_dist'] = raw_df.apply(lambda row: jfk_dist((row.dropoff_latitude, row.dropoff_longitude)), axis=1)
    raw_df['pick_jfk_dist'] = raw_df.apply(lambda row: jfk_dist((row.pickup_latitude, row.pickup_longitude)), axis=1)

#     raw_df['drop_lga_dist'] = raw_df.apply(lambda row: lga_dist((row.dropoff_latitude, row.dropoff_longitude)), axis=1)
#     raw_df['pick_lga_dist'] = raw_df.apply(lambda row: lga_dist((row.pickup_latitude, row.pickup_longitude)), axis=1)

#     raw_df['drop_new_dist'] = raw_df.apply(lambda row: new_dist((row.dropoff_latitude, row.dropoff_longitude)), axis=1)
#     raw_df['pick_new_dist'] = raw_df.apply(lambda row: new_dist((row.pickup_latitude, row.pickup_longitude)), axis=1)
    
#     raw_df['drop_man_dist'] = raw_df.apply(lambda row: man_dist((row.dropoff_latitude, row.dropoff_longitude)), axis=1)
#     raw_df['pick_man_dist'] = raw_df.apply(lambda row: man_dist((row.pickup_latitude, row.pickup_longitude)), axis=1)
 

    raw_df['manhat_dist'] = raw_df.apply(lambda row: manhat_dist((row.pickup_latitude, row.pickup_longitude), (row.dropoff_latitude, row.dropoff_longitude)), axis=1)

    raw_df['abs_lat'] = raw_df.apply(lambda row: abs(row.pickup_latitude - row.dropoff_latitude), axis=1)
    raw_df['abs_lon'] = raw_df.apply(lambda row: abs(row.pickup_longitude - row.dropoff_longitude), axis=1)
    raw_df['euclidean'] = raw_df.apply(lambda row: math.sqrt(row.abs_lat**2 + row.abs_lon**2), axis=1)
#     raw_df['up_price'] = raw_df.apply(lambda row: row.year >= 12, axis=1)

    return raw_df



# Load data
raw_train = pd.read_csv('data/cc_nyc_fare_train_small.csv', parse_dates=['pickup_datetime'])
print('Shape of the raw data: {}'.format(raw_train.shape))

# Transform features using the function you have defined
df_train = process_train_data(raw_train)

# Remove fields that we do not want to train with
X = df_train.drop(['key', 'fare_amount', 'pickup_datetime'], axis=1)

# Extract the value you want to predict
Y = df_train['fare_amount']
print('Shape of the feature matrix: {}'.format(X.shape))

# Build final model with the entire training set
final_model = XGBRegressor()
final_model.fit(X, Y)

# Read and transform test set
raw_test = pd.read_csv('data/cc_nyc_fare_test.csv', parse_dates=['pickup_datetime'])
df_test = process_test_data(raw_test)
X_test = df_test.drop(['key', 'pickup_datetime'], axis=1)

# Make predictions for test set and output a csv file
df_test['predicted_fare_amount'] = final_model.predict(X_test)
df_test[['key', 'predicted_fare_amount']].to_csv('predictions.csv', index=False)



