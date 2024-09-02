import os
import json
import numpy as np
import argparse
from scipy import io

pp = argparse.ArgumentParser(description='Format UCMerced metadata.')
pp.add_argument('--load-path', type=str, default='data/UCMerced_LandUse', help='Path to a directory containing a copy of the UCMerced dataset.')
pp.add_argument('--save-path', type=str, default='data/UCMerced_LandUse', help='Path to output directory.')
args = pp.parse_args()

catName_to_catID = {
    'agricultural' : 0,
    'airplane' : 1, 
    'baseballdiamond' : 2,
    'beach' : 3,
    'buildings' : 4,
    'chaparral' : 5,
    'denseresidential' : 6,
    'forest' : 7,
    'freeway' : 8, 
    'golfcourse':9,
    'harbor':10,
    'intersection':11,
    'mediumresidential':12,
    'mobilehomepark':13,
    'overpass':14,
    'parkinglot':15,
    'river':16,
    'runway':17,
    'sparseresidential':18,
    'storagetanks':19,
    'tenniscourt':20
}

multilabel_metadata = io.loadmat(os.path.join(args.load_path, 'LandUse_multilabels.mat'))


image_list_train = []
image_list_test = []
label_matrix_train = np.zeros((int(2100*0.8), 17))
label_matrix_test = np.zeros((int(2100*0.2), 17))
cnt_train = 0
cnt_test = 0

for clsname in os.listdir(os.path.join(args.load_path, 'Images')):
    for imgname in os.listdir(os.path.join(args.load_path, 'Images', clsname)):
        imgnum = imgname.split('.')[0][-2:]

        imgfullname = os.path.join(args.load_path, 'Images', clsname, imgname)
        imgfullnum = catName_to_catID[clsname] * 100 + int(imgnum)

        if int(imgnum) < 80: #train
            image_list_train.append(imgfullname)
            label_matrix_train[cnt_train] = multilabel_metadata['labels'][:, imgfullnum]
            cnt_train += 1
        else: #test
            image_list_test.append(imgfullname)
            label_matrix_test[cnt_test] = multilabel_metadata['labels'][:, imgfullnum]
            cnt_test += 1

print(cnt_train, cnt_test)

np.save(os.path.join(args.save_path, 'formatted_' + 'train' + '_labels.npy'), label_matrix_train)
np.save(os.path.join(args.save_path, 'formatted_' + 'train' + '_images.npy'), np.array(image_list_train))
np.save(os.path.join(args.save_path, 'formatted_' + 'test' + '_labels.npy'), label_matrix_test)
np.save(os.path.join(args.save_path, 'formatted_' + 'test' + '_images.npy'), np.array(image_list_test))