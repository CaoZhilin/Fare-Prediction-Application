import json
import logging
import os

import pandas as pd
import math
import numpy as np
from flask import Flask, request
from clients.ml_engine import MLEngineClient
from clients.natural_language import NaturalLanguageClient
from clients.google_maps import GoogleMapsClient
from clients.text_to_speech import TextToSpeechClient
from clients.speech_to_text import SpeechToTextClient
import base64
import datetime

app = Flask(__name__)

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
mle_model_name = os.getenv("GCP_MLE_MODEL_NAME")
mle_model_version = os.getenv("GCP_MLE_MODEL_VERSION")

ml_engine_client = MLEngineClient(project_id, mle_model_name, mle_model_version)
google_map_client = GoogleMapsClient()
nl_client = NaturalLanguageClient()
tts_client = TextToSpeechClient()
stt_client = SpeechToTextClient()

def haversine_distance(origin, destination):
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
    raw_df = raw_df.drop(['pickup_datetime'], axis=1)

    return raw_df


@app.route('/')
def index():
    return "Hello"


@app.route('/predict', methods=['POST'])
def predict():
    raw_data_df = pd.read_json(request.data.decode('utf-8'),
                               convert_dates=["pickup_datetime"])
    predictors_df = process_test_data(raw_data_df)
    return json.dumps(ml_engine_client.predict(predictors_df.values.tolist()))


@app.route('/farePrediction', methods=['POST'])
def fare_prediction():
    speech = base64.b64decode(request.data)
    speech_text = stt_client.recognize(speech)
    entities = nl_client.analyze_entities(speech_text)
    res_entities = []
    for entity in entities:
        res_entities.append(entity.name)
        
    print(res_entities)
    
    directions = google_map_client.directions(res_entities[0], res_entities[1])
    start_location = directions[0]['legs'][0]['start_location']
    end_location = directions[0]['legs'][0]['end_location']
    
    data = [{'pickup_datetime': datetime.datetime.now(),
             'pickup_latitude': start_location['lat'],
             'pickup_longitude': start_location['lng'],
             'dropoff_latitude': end_location['lat'],
             'dropoff_longitude': end_location['lng'],
             'passenger_count': 3
            }]
    raw_data_df = pd.DataFrame(data)
    print(raw_data_df)
    predictors_df = process_test_data(raw_data_df)
    fare = ml_engine_client.predict(predictors_df.values.tolist())
    
    text = "Your expected fare from " + res_entities[0] + " to " + res_entities[1] + " is ${:0.2f}".format(fare[0])
    audio_content = tts_client.synthesize_speech(text)
    res = {}

    res['entities'] = res_entities
    res['text'] = text
    res['speech'] = str(base64.b64encode(audio_content).decode("utf-8"))

    res['predicted_fare'] = "{:0.2f}".format(fare[0])
    return json.dumps(res)
    

@app.route('/speechToText', methods=['POST'])
def speech_to_text():
    speech = base64.b64decode(request.data)
    text = stt_client.recognize(speech)
    res = {}
    res['text'] = text
    return json.dumps(res)
    

@app.route('/textToSpeech', methods=['GET'])
def text_to_speech():
    text = request.args.get("text")
    audio_content = tts_client.synthesize_speech(text)
#     print(audio_content)
    res = {}
    res['speech'] = str(base64.b64encode(audio_content).decode("utf-8"))
    return json.dumps(res)

@app.route('/farePredictionVision', methods=['POST'])
def fare_prediction_vision():
    pass


@app.route('/namedEntities', methods=['GET'])
def named_entities():
    text = request.args.get("text")
    entities = nl_client.analyze_entities(text)
    res_entities = []
    for entity in entities:
        res_entities.append(entity.name)
    res = {}
    res['entities'] = res_entities
    return json.dumps(res)


@app.route('/directions', methods=['GET'])
def directions():
    origin = request.args.get("origin")
    destination = request.args.get("destination")
    result = google_map_client.directions(origin, destination)
    start_location = result[0]['legs'][0]['start_location']
    end_location = result[0]['legs'][0]['end_location']
    res = {}
    res['start_location'] = start_location
    res['end_location'] = end_location
    return json.dumps(res)

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    app.run()
