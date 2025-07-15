# AAI-590 Group 9
# Custome Image Classifier that considers cyclical temporal features
# to be updated later

import torch
import torch.nn as nn
import torchvision.models as models

class AnimalClassifier(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        
        # Pretrained ResNet18 backbone
        self.cnn = models.resnet18(pretrained=True)
        cnn_out_dim = self.cnn.fc.in_features # ResNet18 last layer output size (512)
        self.cnn.fc = nn.Linear(cnn_out_dim, num_classes)
        
    def forward(self, image):
        img_vec = self.cnn(image)
        out = img_vec
        return out

class AnimalTemporalClassifier(nn.Module):
    
    def __init__(self, num_classes, proj_dim = 128):
        super().__init__()
        
        # Pretrained ResNet18 backbone
        self.cnn = models.resnet18(pretrained=True)
        cnn_out_dim = self.cnn.fc.in_features # ResNet18 last layer output size (512)
        self.cnn.fc = nn.Identity()  # remove fc layer Output: [batch, 512]
        
        # Projection Layers: convert both image and temporal feature vectors to the same dimension
        # Projection Layer (Image)
        self.img_project = nn.Sequential(
            nn.Linear(cnn_out_dim, proj_dim), # from 512 to 64
            nn.ReLU(),
        )
        # Projection Layer (Temporal)
        self.time_project = nn.Sequential(
            nn.Linear(4, proj_dim),
            nn.ReLU(),
        )

        # Final Classification Layer
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim*2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, features):
        # extract image features and project to fixed size for fusion with temporal feature vector
        img_vec = self.cnn(image) # [batch, 512]
        img_proj = self.img_project(img_vec) # [batch, proj_dim = 64 (default)]

        # project temporal feature vector to fixed size for fusion with projected image feature
        time_proj = self.time_project(features) # [batch, proj_dim = 64 (default)]

        # concatenate projected vectors and feed to classifier layer
        combined = torch.cat([img_proj, time_proj], dim=1) # [batch, proj_dim*2 (128)]
        out = self.classifier(combined)
        
        return out

