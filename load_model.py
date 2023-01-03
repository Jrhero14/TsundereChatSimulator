from sklearn.naive_bayes import GaussianNB
import pickle

def model_load(file_path: str):
    # load the model from disk
    model = pickle.load(open(file_path, 'rb'))
    return model
