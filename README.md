# Instance-Dependent Multi-Label Noise Generation for Multi-Label Remote Sensing Image Classification (IEEE JSTARS)

## Abstract

Multi-label remote sensing image classification is a fundamental task that classifies multiple objects and land covers within an image. However, training deep learning models for this task requires a considerable cost of labeling. While several efforts to reduce labeling costs have been made, they often result in decreased label quality and the inclusion of incorrect (i.e., noisy) labels. To tackle this issue, algorithms for training deep learning models robust to multi-label noise have been proposed in the literature. Nonetheless, the efficacy of these algorithms has been evaluated only under instance-independent multi-label noise, where noise is generated regardless of the individual characteristics and features of each remote sensing image.
In this article, we introduce generating instance-dependent multi-label noise into multi-label remote sensing image datasets for the first time. We leverage a vision-language model with zero-shot prediction capabilities to compute category-wise prediction scores for each image, based on which we generate multi-label noise in an instance-dependent manner. We demonstrate that the proposed instance-dependent multi-label noise is more feasibly generated with respect to individual images compared to traditional instance-independent multi-label noise. We also demonstrate that a more challenging noise scenario is generated, which leads to a more complex decision boundary and stronger overfitting during deep learning model training. Finally, we re-evaluate existing noise-robust training algorithms under the generated instance-dependent multi-label noise and observe that several algorithms exhibit limited robustness against instance-dependent multi-label noise.

## Usage

Tested in Python 3.8.

### Step 1. Install python dependencies.
```
pip install -r requirements.txt
pip install -r requirements_cuda11.txt (if your cuda version is greater than 11.5)
pip install git+https://github.com/openai/CLIP.git
```

### Step 2. Download datasets.

- UCMerced : http://weegee.vision.ucmerced.edu/datasets/landuse.html

- MLRSNet : https://data.mendeley.com/datasets/7j9bv9vwsx/3

- AID : https://github.com/Hua-YS/AID-Multilabel-Dataset

The resulting directory hierarchy should be like this : 

```
data
|--- UCMerced_LandUse
|    |--- Images
|    |--- LandUse_Multilabeled.txt
|    |--- LandUse_Multilabeled.xlsx
|    |--- LandUse_multilabels.mat
|--- MLRSNet
|    |--- Images
|    |--- Labels
|--- AID_multilabel
|    |--- images_tr
|    |--- images_test
|    |--- multilabel.csv
|    |--- multilabel.mat
```

### Step 3. Preprocess datasets.

- UCMerced
```
python preproc/format_ucmerced.py
python clip_ucmerced.py
```

- MLRSNet
```
python preproc/format_mlrsnet.py
python clip_mlrsnet.py
```

- AID
```
python preproc/format_aid.py
python clip_aid.py
```

### Step 4. Run experiments.
```
python main.py --dataset [dataset] \
               --scheme [scheme] \
               --noise_rate [noise_rate] \
```
where [dataset] in {UCMerced, MLRSNet, AID}, [scheme] in {BCE, LCR, RCML, SAT, ELR, JoCoR}, [noise_rate] in {10, 20, 30, 40}.