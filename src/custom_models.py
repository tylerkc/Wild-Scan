# AAI-590 Group 9
# Custome Image Classifier that considers cyclical temporal features
# to be updated later

import torch
import torch.nn as nn
import torchvision.models as models
import logging


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
)
logger = logging.getLogger(__name__)

class CustomClassifier(nn.Module):
    # to add custom classifier later
    def __init__(self, num_classes, name: str = "BuiltFromScratch"):
        super().__init__()

        # assign custom attribute
        self.use_temporal_features = False
        
        # assign a pretrained image classifier backbone
        # Using ResNet50 as the backbone
        logger.debug(f"Initializing CustomClassifier with num_classes={num_classes}")
        self.cnn = models.resnet50(pretrained=True)
        cnn_out_dim = self.cnn.fc.in_features 
        self.cnn.fc = nn.Linear(cnn_out_dim, num_classes)
        
    def forward(self, image):
        img_vec = self.cnn(image)
        out = img_vec
        return out
    
    def __name__(self):
        return f"CustomClassifier(use_temporal_features={self.use_temporal_features})"


class AnimalClassifier(nn.Module):
    
    def __init__(self, num_classes, name: str = "AnimalClassifier_ResNet18"):
        super().__init__()

        # assign custom attribute
        self.use_temporal_features = False
        self.name = name
        
        # Pretrained ResNet18 backbone
        self.cnn = models.resnet18(pretrained=True)
        cnn_out_dim = self.cnn.fc.in_features # ResNet18 last layer output size (512)
        self.cnn.fc = nn.Linear(cnn_out_dim, num_classes)
        
    def forward(self, image):
        img_vec = self.cnn(image)
        out = img_vec
        return out
    
    def __name__(self):
        return self.name
    
class WrapperModel(nn.Module):
    def __init__(self, backbone="resnet50", num_classes=15, pretrained=True, name: str = "WrapperModel"):
        
        #super(WrapperModel, self).__init__()
        super().__init__()

        self.use_temporal_features = False
        self.name = name

        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.model = models.resnet50(weights=weights)
            # Replace the last FC layer
            in_feats = self.model.fc.in_features
            self.model.fc = nn.Linear(in_feats, num_classes)
            self.name = f"{self.name}_ResNet50"
        elif backbone == "vgg16":
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            self.model = models.vgg16(weights=weights)
            # Replace the last classifier layer
            in_feats = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(in_feats, num_classes)
            self.name = f"{self.name}_VGG16"
        else:
            raise ValueError("Unsupported backbone: choose 'resnet50' or 'vgg16'.")

    def forward(self, x):
        return self.model(x)
    
    def __name__(self):
        return self.name

class AnimalTemporalClassifier(nn.Module):
    
    def __init__(self, num_classes, proj_dim = 256, name: str = "AnimalTemporalClassifier_ResNet18"):
        super().__init__()

        # assign custom attributes
        self.use_temporal_features = True
        self.name = name
        
        # Pretrained ResNet18 backbone
        self.cnn = models.resnet18(pretrained=True)
        cnn_out_dim = self.cnn.fc.in_features # ResNet18 last layer output size (512)
        self.cnn.fc = nn.Identity()  # remove fc layer Output: [batch, 512]
        
        # Projection Layers: convert both image and temporal feature vectors to the same dimension
        # Projection Layer (Image)
        self.img_project = nn.Sequential(
            nn.Linear(cnn_out_dim, proj_dim), # from 512 to 256
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
        img_proj = self.img_project(img_vec) # [batch, proj_dim = 256 (default)]

        # project temporal feature vector to fixed size for fusion with projected image feature
        time_proj = self.time_project(features) # [batch, proj_dim = 256 (default)]

        # concatenate projected vectors and feed to classifier layer
        combined = torch.cat([img_proj, time_proj], dim=1) # [batch, proj_dim*2 (512)]
        out = self.classifier(combined)
        
        return out
    
    def __name__(self):
        return self.name
