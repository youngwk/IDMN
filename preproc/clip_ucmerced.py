import datasets
import numpy as np
from PIL import Image
import torch
import os
import clip

UCMERCED_CATEGORY = ['airplane', 'bare soil', 'buildings', 'cars', 'chaparral', 'court', 'dock', 'field', 'grass', 'mobile home', 'pavement', 'sand', 'sea', 'ship', 'tanks', 'trees', 'water']

class Config:
    pass
args = Config()
args.dataset = 'UCMerced'
args.image_size = 256
args.num_classes = 17
args.path_to_dataset = 'data/UCMerced_LandUse'

device = 'cuda'
model, preprocess = clip.load('ViT-B/32', device=device)

text_inputs = torch.cat([clip.tokenize(f"an aerial image of a {c}") for c in UCMERCED_CATEGORY]).to('cuda')


dataset = datasets.get_dataset(args)

print(np.sum(dataset['train'].label_matrix))

with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features @ text_features.T

np.save('data/UCMerced_LandUse/clip_adjacency.npy', similarity.cpu().numpy())
    


fp_clip = np.zeros((len(dataset['train']), args.num_classes))
num_fp = 0
fn_clip = np.zeros((len(dataset['train']), args.num_classes))
num_fn = 0

clip_logits_matrix = np.zeros((len(dataset['train']), args.num_classes))

for idx in range(len(dataset['train'])):
    image_path = dataset['train'].image_paths[idx]
    label = dataset['train'].label_matrix[idx]
    pos_idx = np.nonzero(label)[0]
    neg_idx = np.nonzero(label == 0)[0]

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        _, logits_per_text = model(image, text_inputs)
        logits_per_text = logits_per_text.squeeze().cpu()

        clip_logits_matrix[idx] = logits_per_text

np.save('data/UCMerced_LandUse/clip_logits.npy', clip_logits_matrix)

