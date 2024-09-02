import os
import json
import numpy as np
import argparse
from scipy import io

pp = argparse.ArgumentParser(description='Format AID metadata.')
pp.add_argument('--load-path', type=str, default='data/AID_multilabel', help='Path to a directory containing a copy of the AID dataset.')
pp.add_argument('--save-path', type=str, default='data/AID_multilabel', help='Path to output directory.')
args = pp.parse_args()

multilabel_metadata = io.loadmat(os.path.join(args.load_path, 'multilabel.mat'))

scenenames = os.listdir('data/AID_multilabel/images_tr/')
scenenames.sort()
scenenames = [x.lower() for x in scenenames]

image_list_train = []
image_list_test = []
label_matrix_train = np.zeros((2400, 17))
label_matrix_test = np.zeros((600, 17))
cnt_train = 0
cnt_test = 0

for clsname in os.listdir(os.path.join(args.load_path, 'images_tr')):
    for imgname in os.listdir(os.path.join(args.load_path, 'images_tr', clsname)):
        imgnum = int(imgname.split('.')[0].split('_')[1])
        if clsname == 'Park':
            imgnum -= 100

        imgfullname = os.path.join(args.load_path, 'images_tr', clsname, imgname)
        imgfullnum = scenenames.index(clsname.lower()) * 100 + imgnum - 1

        image_list_train.append(imgfullname)
        label_matrix_train[cnt_train] = multilabel_metadata['labels'][:, imgfullnum]
        cnt_train += 1
        
        
for clsname in os.listdir(os.path.join(args.load_path, 'images_test')):
    for imgname in os.listdir(os.path.join(args.load_path, 'images_test', clsname)):
        imgnum = int(imgname.split('.')[0].split('_')[1])
        if clsname == 'Park':
            imgnum -= 100

        imgfullname = os.path.join(args.load_path, 'images_test', clsname, imgname)
        imgfullnum = scenenames.index(clsname.lower()) * 100 + imgnum - 1

        image_list_test.append(imgfullname)
        label_matrix_test[cnt_test] = multilabel_metadata['labels'][:, imgfullnum]
        cnt_test += 1

print(cnt_train, cnt_test)

np.save(os.path.join(args.save_path, 'formatted_' + 'train' + '_labels.npy'), label_matrix_train)
np.save(os.path.join(args.save_path, 'formatted_' + 'train' + '_images.npy'), np.array(image_list_train))
np.save(os.path.join(args.save_path, 'formatted_' + 'test' + '_labels.npy'), label_matrix_test)
np.save(os.path.join(args.save_path, 'formatted_' + 'test' + '_images.npy'), np.array(image_list_test))
