
from PIL import Image
from shutil import copyfile
from data import recursive_listdir


pic_dir = '/Users/sasha/scikit_learn_data/lfw_home/lfw_funneled'
pic_proc_dir_1 = '/Users/sasha/Documents/Python/ml/faces/data'
pic_proc_dir_2 = '/Users/sasha/Documents/Python/ml/faces/data2'

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

for i, (s, d) in enumerate(recursive_listdir(pic_proc_dir_1, pic_proc_dir_2)):
    if i > 1:
        break
    if s.endswith('n.jpg'):
        copyfile(s, d)
        print(s, '->', d)

print('processed count:', i)
