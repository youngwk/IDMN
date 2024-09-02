import pandas as pd
import os
import os.path
import numpy as np

MLRSNET_CATEGORY = ['airplane','airport','bare soil','baseball diamond','basketball court','beach','bridge','buildings','cars','chaparral','cloud','containers','crosswalk','dense residential area','desert','dock','factory','field','football field','forest','freeway','golf course','grass','greenhouse','gully','habor','intersection','island','lake','mobile home','mountain','overpass','park','parking lot','parkway','pavement','railway','railway station','river','road','roundabout','runway','sand','sea','ships','snow','snowberg','sparse residential area','stadium','swimming pool','tanks','tennis court','terrace','track','trail','transmission tower','trees','water','wetland','wind turbine']

RandomGenerator = np.random.RandomState(seed=999)
root_path = './data/MLRSNet'
label_files = os.listdir(os.path.join(root_path, 'Labels'))

image_paths_train = []
labels_train = []
image_paths_test = []
labels_test = []

num_train = 0
num_test = 0

for label_file in label_files:
    label = pd.read_csv(os.path.join(root_path, 'Labels', label_file))
    num_images = label.shape[0]
    for i in range(num_images):
        row = label.iloc[i].values
        if i < int(num_images * 0.8):
            image_paths_train.append(os.path.join(root_path, 'Images', row[0]))
            labels_train.append(row[1:].astype(np.float32))
            num_train += 1
        else:
            image_paths_test.append(os.path.join(root_path, 'Images', row[0]))
            labels_test.append(row[1:].astype(np.float32))
            num_test += 1

print(num_train, num_test)
    
labels_train = np.stack(labels_train)
labels_test = np.stack(labels_test)

np.save(os.path.join(root_path, 'formatted_' + 'train' + '_labels.npy'), labels_train)
np.save(os.path.join(root_path, 'formatted_' + 'test' + '_labels.npy'), labels_test)

np.save(os.path.join(root_path, 'formatted_' + 'train' + '_images.npy'), image_paths_train)
np.save(os.path.join(root_path, 'formatted_' + 'test' + '_images.npy'), image_paths_test)