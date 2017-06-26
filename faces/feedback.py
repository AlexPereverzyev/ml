
import os
from PIL import Image
from shutil import copyfile, move
from data import recursive_listdir


negative = 'matches'
positive = 'positive'
train_data = 'data'

# add negative matches to training data set
for i, (s, d) in enumerate(recursive_listdir(negative, train_data)):
    print(s, '->', d)
    copyfile(s, d)
print('processed count:', i)

# add positive matches to training data set
# start, count = 0, 1000
# for i, (s, d) in enumerate(recursive_listdir(positive, train_data)):
#     file_idx = int(os.path.basename(s).split('.')[0])
#     if start <= file_idx <= (start + count):
#         print(s, '->', d)
#         copyfile(s, d)
# print('processed count:', i)

# crop and convert images
# pic_dir = '/Users/sasha/scikit_learn_data/lfw_home/lfw_funneled'
# crop_size = (120, 120)
# size = (70, 70)
# for i, (s, d) in enumerate(recursive_listdir(pic_dir, positive)):
#     img = Image.open(s)
#     l = r = (img.size[0] - crop_size[0]) / 2
#     t = b = (img.size[1] - crop_size[1]) / 2
#     r += crop_size[0]
#     b += crop_size[1]
#     img = img.convert('L')
#     img = img.crop((l, t, r, b))
#     img = img.resize(size)
#     img.save(d)
# print('processed count:', i)
