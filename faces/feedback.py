
import os
from PIL import Image
from shutil import copyfile, move
from data import recursive_listdir


temp = '/Users/sasha/Documents/Python/ml/faces/matches'
pic_dir = '/Users/sasha/scikit_learn_data/lfw_home/lfw_funneled'
pic_proc_dir_1 = '/Users/sasha/Documents/Python/ml/faces/data'
pic_proc_dir_2 = '/Users/sasha/Documents/Python/ml/faces/positive'

# crop and convert images
# crop_size = (120, 120)
# size = (70, 70)
# for i, (s, d) in enumerate(recursive_listdir(pic_dir, pic_proc_dir_2)):
#     img = Image.open(s)
#     l = r = (img.size[0] - crop_size[0]) / 2
#     t = b = (img.size[1] - crop_size[1]) / 2
#     r += crop_size[0]
#     b += crop_size[1]
#     img = img.convert('L')
#     img = img.crop((l, t, r, b))
#     img = img.resize(size)
#     img.save(d)

# move all negative matches from temp dir to training data dir
for i, (s, d) in enumerate(recursive_listdir(temp, pic_proc_dir_1)):
    print(s, '->', d)
    move(s, d)

# print('processed count:', i)

# add more positive matches to training data dir
# start, count = 1075, i + 1
# for i, (s, d) in enumerate(recursive_listdir(pic_proc_dir_2, pic_proc_dir_1)):
#     file_idx = int(os.path.basename(s).split('.')[0])
#     if start <= file_idx <= count:
#         print(s, '->', d)
#         # copyfile(s, d)

# print('processed count:', i)
