
import os
from PIL import Image
from matching import MatchDetector
from persistence import ModelStore
from data import recursive_listdir


iteration = 3
model_name = 'pca_svc_2.pkl'
models_path = 'models'
targets_path = 'targets'
matches_path = 'matches'

store = ModelStore(models_path)
clf = store.load(model_name)
detector = MatchDetector(clf)

for img_path, _ in recursive_listdir(targets_path, ''):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img = Image.open(img_path)
    for i, (c, b, r) in enumerate(detector.find_all(img)):
        match_file = '{0}_{1}_{2}_n.jpg'.format(img_name, iteration, i + 1)
        match_path = os.path.join(matches_path, match_file)
        match = img.crop(b)
        match.save(match_path)
        print(match_file, '{0:.2f}'.format(c), b)
