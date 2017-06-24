
import os
import datetime

from sklearn.externals import joblib


class ModelStore(object):
    def __init__(self, models_path):
        self.models_path = models_path

    def save(self, model, print_name=True, id=None):
        # only accepts pipes for now
        model_name = '_'.join([k for k in sorted(model.named_steps)])
        if id is None:
            id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        model_path = os.path.join(
            self.models_path, '{0}_{1}.pkl'.format(model_name, id))
        joblib.dump(model, model_path)
        if print_name:
            print('Model saved to: ', model_path)

    def load(self, model_name):
        model_path = os.path.join(self.models_path, model_name)
        model = joblib.load(model_path)
        return model
