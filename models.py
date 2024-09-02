import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    
    def forward(self, feature_map):
        return F.adaptive_avg_pool2d(feature_map, 1).squeeze(-1).squeeze(-1)

class ImageClassifier(torch.nn.Module):
    def __init__(self, arch, num_classes):
        super(ImageClassifier, self).__init__()
        self.arch = arch
        self.num_classes = num_classes

        feature_extractor = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        feat_dim = 2048
        
        self.feature_extractor1 = torch.nn.Sequential(*list(feature_extractor.children())[:5])
        self.feature_extractor2 = torch.nn.Sequential(*list(feature_extractor.children())[5:-2])
        
        self.avgpool = GlobalAvgPool2d()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        intermediate_feats = self.feature_extractor1(x)
        feats = self.feature_extractor2(intermediate_feats)
        pooled_feats = self.avgpool(feats)
        logits = self.fc(pooled_feats)
        return logits
    
    def get_feats_and_logits(self, x):
        intermediate_feats = self.feature_extractor1(x)
        feats = self.feature_extractor2(intermediate_feats)
        pooled_feats = self.avgpool(feats)
        logits = self.fc(pooled_feats)
        return logits, intermediate_feats
    