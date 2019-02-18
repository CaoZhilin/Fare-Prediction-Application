"""
Local server test.
"""
import requests
import pandas as pd
endpoint = "http://localhost:5000"
# endpoint = "https://linen-option-222503.appspot.com"
data = pd.read_csv('input.csv').to_json(orient='records')
print(requests.post('{}/predict'.format(endpoint), data=data).json())

direction = {
    'origin':'Pennsylvania Station',
    'destination': 'Times Square'
}

entity = {
    'text': 'I would like to go from Central Park Zoo to Bronx Zoo.'
}

text = {
    'text': 'American Museum of Natural History and Bryant Park'
}

print(requests.get('{}/textToSpeech'.format(endpoint), params=text).json())
print(requests.get('{}/namedEntities'.format(endpoint), params=entity).json())
print(requests.get('{}/directions'.format(endpoint), params=direction).json())