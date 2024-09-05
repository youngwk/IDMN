import numpy as np
import torch
import models
import os
import torch.nn.functional as F
from instrumentation import compute_metrics
import losses
import datasets

def run_train(args):
    dataset = datasets.get_dataset(args)

    if args.noise_rate != 0:
        dataset['train'].inject_noise(args)

    dataloader = {}
    for phase in ['train', 'test']:
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase], 
            batch_size = args.bsize,
            shuffle = phase == 'train',
            sampler = None,
            num_workers = args.num_workers,
            drop_last = False,
            pin_memory = True
        )

    model1 = models.ImageClassifier(args.num_classes)
    model2 = models.ImageClassifier(args.num_classes)
    
    optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader['train'])*5)

    device = f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu'
    model1.to(device)
    model2.to(device)
    args.device = device

    criterion = losses.get_criterion(args)
    
    for epoch in range(1, args.num_epochs+1):
        print(f'Epoch {epoch}')
        
        model1.train()
        model2.train()
        
        with torch.set_grad_enabled(True):
            for image, label, index in dataloader['train']:
                # Move data to GPU
                image = image.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits1 = model1(image)
                logits2 = model2(image)
                if logits1.dim() == 1:
                    logits1 = torch.unsqueeze(logits1, 0)
                    logits2 = torch.unsqueeze(logits2, 0)

                preds1 = torch.sigmoid(logits1)
                preds2 = torch.sigmoid(logits2)

                loss = criterion(logits1, logits2, label)
                loss.backward()
                optimizer.step()
                scheduler.step()
        
        criterion.end_of_epoch()

       
    model1.eval()
    model2.eval()
    y_pred1 = np.zeros((len(dataset['test']), args.num_classes))
    y_pred2 = np.zeros((len(dataset['test']), args.num_classes))
    y_true = np.zeros((len(dataset['test']), args.num_classes))
    batch_stack = 0
    
    with torch.set_grad_enabled(False):
        for image, label, index in dataloader['test']:
            # Move data to GPU
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits1 = model1(image)
            logits2 = model2(image)
            if logits1.dim() == 1:
                logits1 = torch.unsqueeze(logits1, 0)
                logits2 = torch.unsqueeze(logits2, 0)

            preds1 = torch.sigmoid(logits1)
            preds2 = torch.sigmoid(logits2)

            
            preds1_np = preds1.cpu().numpy()
            preds2_np = preds2.cpu().numpy()
            this_batch_size = preds1_np.shape[0]
            y_pred1[batch_stack : batch_stack+this_batch_size] = preds1_np
            y_pred2[batch_stack : batch_stack+this_batch_size] = preds2_np
            y_true[batch_stack : batch_stack+this_batch_size] = label.cpu().numpy()
            batch_stack += this_batch_size

    metrics1 = compute_metrics(y_pred1, y_true)
    metrics2 = compute_metrics(y_pred2, y_true)
    
    map1 = metrics1['map']
    map2 = metrics2['map']

    hamming1 = metrics1['hamming']
    hamming2 = metrics2['hamming']

    
    if map1 > map2:
        map = map1
        hamming = hamming1
    else:
        map = map2
        hamming = hamming2

    print(f"test mAP macro {map:.3f}, hamming loss {hamming:.3f}")


    print('Training procedure completed!')



if __name__ == "__main__":
    run_train()