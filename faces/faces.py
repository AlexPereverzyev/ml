
from preprocess import decompose
from matching import MatchDetector
from persistence import ModelStore

image_path = 'targets/sample_b.jpg'
models_path = 'models'
model_name = 'pca_svc_20170619-124909.pkl'

store = ModelStore(models_path)
clf = store.load(model_name)
detector = MatchDetector(clf)

c = 0
for i, (r, s, b) in enumerate(decompose(image_path, step=20)):
    if detector.is_match(r):
        c += 1
        print(c, s, b)
        # r.save('temp/b_{0}_n.jpg'.format(c))
        # r.save('targets/sample_a_face_{0}.jpg'.format(c))
print('Total Checks:', i + 1)
