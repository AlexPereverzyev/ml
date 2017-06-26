
import os
from preprocess import decompose
from matching import MatchDetector
from persistence import ModelStore
from data import recursive_listdir


iteration = 2
model_name = 'pca_svc_1.pkl'
models_path = 'models'
targets_path = 'targets'
matches_path = 'matches'

store = ModelStore(models_path)
clf = store.load(model_name)
detector = MatchDetector(clf)

for img_path, d in recursive_listdir(targets_path, ''):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    c = 0
    for i, (r, s, b) in enumerate(decompose(img_path, step=15)):
        is_face, confidence = detector.match(r)
        if is_face:
            c += 1
            match_file = '{0}_{1}_{2}_n.jpg'.format(img_name, iteration, c)
            match_path = os.path.join(matches_path, match_file)
            r.save(match_path)
            print(match_file, '{0:.2f} {1:.2f}'.format(confidence, s), b)
    print(img_name, ': processed fragments:', i + 1)
