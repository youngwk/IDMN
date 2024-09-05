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

    model = models.ImageClassifier(args.num_classes)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader['train'])*5)
    device = f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    args.device = device

    criterion = losses.get_criterion(args)

    for epoch in range(1, args.num_epochs+1):
        print(f'Epoch {epoch}')
        model.train()
        with torch.set_grad_enabled(True):
            for image, label, index in dataloader['train']:
                # Move data to GPU
                image = image.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(image)
                if logits.dim() == 1:
                    logits = torch.unsqueeze(logits, 0)

                preds = torch.sigmoid(logits)

                
                loss = criterion(logits, label, index)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
        criterion.end_of_epoch()
    
    model.eval()
    y_pred = np.zeros((len(dataset['test']), args.num_classes))
    y_true = np.zeros((len(dataset['test']), args.num_classes))
    batch_stack = 0
            
    with torch.set_grad_enabled(False):
        for image, label, index in dataloader['test']:
            # Move data to GPU
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(image)
            if logits.dim() == 1:
                logits = torch.unsqueeze(logits, 0)

            preds = torch.sigmoid(logits)

            
            preds_np = preds.cpu().numpy()
            this_batch_size = preds_np.shape[0]
            y_pred[batch_stack : batch_stack+this_batch_size] = preds_np
            y_true[batch_stack : batch_stack+this_batch_size] = label.cpu().numpy()
            batch_stack += this_batch_size

    metrics = compute_metrics(y_pred, y_true)
    map = metrics['map']
    hamming = metrics['hamming']

    print(f"test mAP macro {map:.3f}, hamming loss {hamming:.3f}")

        
   
    print('Training procedure completed!')


if __name__ == "__main__":
    run_train()