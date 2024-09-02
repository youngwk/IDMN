import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def get_criterion(args):
    if args.scheme == 'BCE':
        return BCE()
    elif args.scheme == 'ELR':
        return ELR(args)
    elif args.scheme == 'SAT':
        return SAT(args)
    elif args.scheme == 'LCR':
        return LCR(args)
    elif args.scheme == 'JoCoR':
        return JoCoR(args)
    elif args.scheme == 'RCML':
        return RCML(args)

class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()
    def end_of_epoch(self):
        pass
    def forward(self, logits, labels, index):
        return F.binary_cross_entropy_with_logits(logits, labels)

class ELR(nn.Module):
    def __init__(self, args):
        super(ELR, self).__init__()
        self.lam = args.lam1
    def end_of_epoch(self):
        pass
    def forward(self, logits, labels, index):
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        preds = torch.sigmoid(logits)
        preds = torch.clamp(preds, 1e-4, 1.0-1e-4)
        preds = preds.data.detach()
        reg = torch.mean(torch.log(1 - preds * labels))
        return bce + self.lam * reg
    
class SAT(nn.Module):
    def __init__(self, args):
        super(SAT, self).__init__()
        self.device = args.device
        self.targets = torch.zeros((args.num_train, args.num_classes))
        self.epoch = 1
        self.Es = args.Es
    
    def end_of_epoch(self):
        self.epoch += 1
    
    def forward(self, logits, labels, index):
        if self.epoch == 1:
            self.targets[index] = labels.clone().detach().cpu()
        if self.epoch > self.Es:
            # update the target
            preds = torch.sigmoid(logits)
            targets = self.targets[index].clone().to(self.device)
            targets = 0.9 * targets + 0.1 * preds
            self.targets[index] = targets.clone().detach().cpu()
        else:
            targets = labels
        return F.binary_cross_entropy_with_logits(logits, targets)
    
class LCR(nn.Module):
    def __init__(self, args):
        super(LCR, self).__init__()
        word_embedding_path = f'wordembedding/{args.dataset}_glove.npy'
        self.word_embedding = torch.from_numpy(np.load(word_embedding_path)).to(args.device)
        self.lam = args.lam1

    def end_of_epoch(self):
        pass

    def forward(self, logits, labels, index):
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        preds = torch.round(torch.sigmoid(logits))
        preds = preds.data.detach()
        num_images = logits.shape[0]
        num_classes = logits.shape[1]

        lcr = 0
        for i in range(num_images):
            matrix1 = preds[i].repeat(num_classes, 1) * self.word_embedding
            matrix2 = torch.outer(preds[i], preds[i])
            lcr += torch.mean(torch.sqrt(torch.sum(torch.square(matrix1 - matrix2), dim=1)))
            
        return bce + self.lam * lcr / num_images
    
class JoCoR(nn.Module):
    def __init__(self, args):
        super(JoCoR, self).__init__()
        self.lam = args.lam3
        self.Tk = args.Tk
        self.tau = args.tau
        self.smallloss_rate = 1

    def end_of_epoch(self):
        if self.smallloss_rate != 1 - self.tau:
            self.smallloss_rate -= self.tau / self.Tk


    def forward(self, logits1, logits2, labels):
        bce_loss = F.binary_cross_entropy_with_logits(logits1, labels, reduction='none') + F.binary_cross_entropy_with_logits(logits2, labels, reduction='none')
        
        preds1 = torch.sigmoid(logits1)
        preds2 = torch.sigmoid(logits2)
        preds1 = torch.clamp(preds1, 1e-4, 1.0-1e-4)#.detach()
        preds2 = torch.clamp(preds2, 1e-4, 1.0-1e-4)#.detach()
        # kl_loss = preds1 * torch.log(preds1/preds2) + (1 - preds1) * torch.log((1 - preds1)/(1-preds2)) + preds2 * torch.log(preds2/preds1) + (1 - preds2) * torch.log((1-preds2)/(1-preds1))
        kl_loss = F.kl_div(preds1, preds2, reduction='none') + F.kl_div(1 - preds1, 1 - preds2, reduction='none') + F.kl_div(preds2, preds1, reduction='none') + F.kl_div(1 - preds2, 1 - preds1, reduction='none')
        total_loss = (1 - self.lam) * bce_loss + self.lam * kl_loss # matrix
        
        small_loss = torch.sort(total_loss.flatten()).values[:int(self.smallloss_rate * labels.shape[0] * labels.shape[1])]

        return torch.mean(small_loss)
    
class RCML(nn.Module):
    def __init__(self, args):
        super(RCML, self).__init__()
        self.sigma = 10000
        self.alpha = args.alpha
        self.lam1 = args.lam1
        self.lam2 = args.lam2
        self.swap_rate = args.swap_rate
        
    def end_of_epoch(self):
        pass

    def mmd(self, X, Y):
        X = torch.reshape(X, [X.shape[0], -1])
        Y = torch.reshape(Y, [Y.shape[0], -1])

        XX = torch.mm(X, X.T)
        XY = torch.mm(X, Y.T)
        YY = torch.mm(Y, Y.T)

        X_sqnorms = torch.diagonal(XX)
        Y_sqnorms = torch.diagonal(YY)

        r = lambda x: torch.unsqueeze(x, 0)
        c = lambda x: torch.unsqueeze(x, 1)

        K_XX = torch.exp(-(1./self.sigma) * (-2. * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY = torch.exp(-(1./self.sigma) * (-2. * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY = torch.exp(-(1./self.sigma) * (-2. * YY + c(Y_sqnorms) + r(Y_sqnorms)))

        return torch.mean(K_XX) - 2. * torch.mean(K_XY) + torch.mean(K_YY)

    def grouplasso(self, preds, labels):
        inverted_labels = torch.logical_not(labels).float()
        selected_examples = torch.unsqueeze(labels, 1) * torch.unsqueeze(inverted_labels, 2)

        ranking_error = torch.square(F.relu(1 + 2 * (torch.unsqueeze(preds, 2) - torch.unsqueeze(preds, 1))))
        selected_ranking_error = selected_examples * ranking_error
        
        groups_fn = torch.sqrt(torch.sum(selected_ranking_error, dim=2) + 1e-10)
        ranking_loss_fn = torch.sum(groups_fn, dim=1)

        groups_fp = torch.sqrt(torch.sum(selected_ranking_error, dim=1) + 1e-10)
        ranking_loss_fp = torch.sum(groups_fp, dim=1)

        ranking_loss = self.alpha * ranking_loss_fn + (1 - self.alpha) * ranking_loss_fp
        return ranking_loss
    
    def forward(self, logits1, logits2, featuremap1, featuremap2, labels):
        batch_size = labels.shape[0]
        
        preds1 = torch.sigmoid(logits1)
        preds2 = torch.sigmoid(logits2)
        lasso1 = self.grouplasso(preds1, labels).detach()
        lasso2 = self.grouplasso(preds2, labels).detach()

        bce_loss1 = F.binary_cross_entropy_with_logits(logits1, labels, reduction='none')
        bce_loss2 = F.binary_cross_entropy_with_logits(logits2, labels, reduction='none')

        low_loss_idx1 = torch.argsort(lasso1)[:int(batch_size * self.swap_rate)]
        low_loss_idx2 = torch.argsort(lasso2)[:int(batch_size * self.swap_rate)]

        consistency_loss1 = self.mmd(logits1, logits2.detach())
        consistency_loss2 = self.mmd(logits1.detach(), logits2)
        disparity_loss1 = self.mmd(featuremap1, featuremap2.detach())
        disparity_loss2 = self.mmd(featuremap1.detach(), featuremap2)

        final_loss1 = torch.mean(bce_loss1[low_loss_idx2]) + self.lam1 * consistency_loss1 - self.lam2 * disparity_loss1
        final_loss2 = torch.mean(bce_loss2[low_loss_idx1]) + self.lam1 * consistency_loss2 - self.lam2 * disparity_loss2
        return final_loss1, final_loss2    
    