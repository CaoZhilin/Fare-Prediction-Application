from google.cloud import storage
import pandas as pd
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from hypertune import HyperTune
import argparse
import os
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge,BayesianRidge  
from sklearn.cluster import MiniBatchKMeans  
from xgboost.sklearn import XGBRegressor

# ==========================
# ==== Define Variables ====
# ==========================
SAMPLE_PROB = 1.0
random.seed(100)
OUTPUT_BUCKET_ID = 'ml-fare-prediction-gs'
DATA_BUCKET_ID = 'p42ml'
TRAIN_FILE = 'data/cc_nyc_fare_train.csv'


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


# =====================================
# ==== Define data transformations ====
# =====================================

def process_train_data(raw_df):
    """
    :param raw_df: the DataFrame of the raw training data
    :return:  a DataFrame with the predictors created
    """
    # drop null
    raw_df.dropna(inplace=True)
    # get distance
    raw_df['distance'] = raw_df.apply(lambda row: haversine_distance((row.pickup_latitude, row.pickup_longitude), (row.dropoff_latitude, row.dropoff_longitude)), axis=1)
    # drop outliner data
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

    # get direction
    df['direction'] = df.apply(lambda row: direction((row.pickup_latitude, row.pickup_longitude), (row.dropoff_latitude, row.dropoff_longitude)), axis=1)
    
    # convert timestamp to year, hour, day
    df['year'] = df.apply(lambda row: row.pickup_datetime.year, axis=1)
    df['hour'] = df.apply(lambda row: row.pickup_datetime.hour, axis=1)
    df['day'] = df.apply(lambda row: row.pickup_datetime.weekday(), axis=1)

    # distance from jfk airport
    df['drop_jfk_dist'] = df.apply(lambda row: jfk_dist((row.dropoff_latitude, row.dropoff_longitude)), axis=1)
    df['pick_jfk_dist'] = df.apply(lambda row: jfk_dist((row.pickup_latitude, row.pickup_longitude)), axis=1) 
    
    # manhatten distance
    df['manhat_dist'] = df.apply(lambda row: manhat_dist((row.pickup_latitude, row.pickup_longitude), (row.dropoff_latitude, row.dropoff_longitude)), axis=1)
    
    # abs lat and lon
    df['abs_lat'] = df.apply(lambda row: abs(row.pickup_latitude - row.dropoff_latitude), axis=1)
    df['abs_lon'] = df.apply(lambda row: abs(row.pickup_longitude - row.dropoff_longitude), axis=1)
    
    # euclidean distance
    df['euclidean'] = df.apply(lambda row: math.sqrt(row.abs_lat**2 + row.abs_lon**2), axis=1)

    df = df.drop(['fare_dist'], axis=1)

    return df

def jfk_dist(coordinate):
    jfk_dist = haversine_distance((40.6413, -73.7781), coordinate)
    return jfk_dist

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
    raw_df['direction'] = raw_df.apply(lambda row: direction((row.pickup_latitude, row.pickup_longitude), (row.dropoff_latitude, row.dropoff_longitude)), axis=1)
    raw_df['year'] = raw_df.apply(lambda row: row.pickup_datetime.year, axis=1)
    raw_df['hour'] = raw_df.apply(lambda row: row.pickup_datetime.hour, axis=1)
    raw_df['day'] = raw_df.apply(lambda row: row.pickup_datetime.weekday(), axis=1)

    raw_df['drop_jfk_dist'] = raw_df.apply(lambda row: jfk_dist((row.dropoff_latitude, row.dropoff_longitude)), axis=1)
    raw_df['pick_jfk_dist'] = raw_df.apply(lambda row: jfk_dist((row.pickup_latitude, row.pickup_longitude)), axis=1)

    raw_df['manhat_dist'] = raw_df.apply(lambda row: manhat_dist((row.pickup_latitude, row.pickup_longitude), (row.dropoff_latitude, row.dropoff_longitude)), axis=1)

    raw_df['abs_lat'] = raw_df.apply(lambda row: abs(row.pickup_latitude - row.dropoff_latitude), axis=1)
    raw_df['abs_lon'] = raw_df.apply(lambda row: abs(row.pickup_longitude - row.dropoff_longitude), axis=1)
    raw_df['euclidean'] = raw_df.apply(lambda row: math.sqrt(row.abs_lat**2 + row.abs_lon**2), axis=1)

    return raw_df

if __name__ == '__main__':
    # ===========================================
    # ==== Download data from Google Storage ====
    # ===========================================
    print('Downloading data from google storage')
    print('Sampling {} of the full dataset'.format(SAMPLE_PROB))
    input_bucket = storage.Client().bucket(DATA_BUCKET_ID)
    output_bucket = storage.Client().bucket(OUTPUT_BUCKET_ID)
    input_bucket.blob(TRAIN_FILE).download_to_filename('train.csv')

    raw_train = pd.read_csv('train.csv', parse_dates=["pickup_datetime"],
                            skiprows=lambda i: i > 0 and random.random() > SAMPLE_PROB)

    print('Read data: {}'.format(raw_train.shape))

    # =============================
    # ==== Data Transformation ====
    # =============================
    df_train = process_train_data(raw_train)

    # Prepare feature matrix X and labels Y
    X = df_train.drop(['key', 'fare_amount', 'pickup_datetime'], axis=1)
    Y = df_train['fare_amount']
    X_train, X_eval, y_train, y_eval = train_test_split(X, Y, test_size=0.33)
    print('Shape of feature matrix: {}'.format(X_train.shape))

    # ======================================================================
    # ==== Improve model performance with hyperparameter tuning ============
    # ======================================================================

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',  # MLE passes this in by default
        required=True
    )

    parser.add_argument(
        '--max_depth',
        default=6,
        type=int
    )

    parser.add_argument(
        '--min_child_weight',
        default=1,
        type=int
    )
    
    parser.add_argument(
        '--eta',
        default=0.5,
        type=float
    )
    
    parser.add_argument(
        '--subsample',
        default=0.9,
        type=float
    )

    args = parser.parse_args()
    params = {
        'max_depth': args.max_depth,
        'min_child_weight': args.min_child_weight,
        'eta': args.eta,
        'subsample': args.subsample,
        'lambda': 1,
        'booster': 'gbtree',
        'silent': 1,
        'eval_metric': 'rmse'
    }

    # ===============================================
    # ==== Evaluate performance against test set ====
    # ===============================================
    # Create DMatrix for XGBoost from DataFrames
    d_matrix_train = xgb.DMatrix(X_train, y_train)
    d_matrix_eval = xgb.DMatrix(X_eval)
    model = xgb.train(params, d_matrix_train)
    y_pred = model.predict(d_matrix_eval)
    rmse = math.sqrt(mean_squared_error(y_eval, y_pred))
    print('RMSE: {:.3f}'.format(rmse))

    # Return the score back to HyperTune to inform the next iteration
    # of hyperparameter search
    hpt = HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='nyc_fare',
        metric_value=rmse)

    # ============================================
    # ==== Upload the model to Google Storage ====
    # ============================================
    JOB_NAME = os.environ['CLOUD_ML_JOB_ID']
    TRIAL_ID = os.environ['CLOUD_ML_TRIAL_ID']
    model_name = 'model.bst'
    model.save_model(model_name)
    blob = output_bucket.blob('{}/{}_rmse{:.3f}_{}'.format(
        JOB_NAME,
        TRIAL_ID,
        rmse,
        model_name))
    blob.upload_from_filename(model_name)
