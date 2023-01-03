from sklearn.naive_bayes import GaussianNB
import pandas as pd
from nlp_utils import tokenize, bag_of_words
import pickle
import json, random

with open('tsundere.json', 'r') as json_data:
    intents = json.load(json_data)

config_path = 'pkl/config.sav'
def responseChat(sentence: str, model: GaussianNB):
    config_data = pickle.load(open(config_path, 'rb'))
    all_words = config_data['all_words']
    tags = config_data['tags']

    sentence = tokenize(sentence=sentence)
    X = bag_of_words(sentence, all_words)
    final_X = pd.DataFrame(X.reshape(1, X.shape[0]))

    predictTag = model.predict(final_X.values)
    tag = tags[predictTag[0]]

    for intent in intents['intents']:
        if tag == intent["tag"]:
            respon = random.choice(intent['responses'])
            return respon