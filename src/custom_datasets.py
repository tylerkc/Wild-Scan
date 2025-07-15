# AAI-590 Group 9
# S3ImageFeatureDataset class with cyclical temporal features
# to be updated later

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import s3fs
from torchvision import transforms
import sys

class S3ImageWithTimeFeatureDataset(Dataset):
    
    def __init__(self, s3_csv_path, s3_label2idx_path = None):

        # load meta data csv to process into pandas df
        self.df = pd.read_csv(s3_csv_path)

        # transform an input Image into a Tensor then normalize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # initialize s3fs access 
        self.fs = s3fs.S3FileSystem()

        # if label2idx path is given, it is assumed that label encoding is required
        if s3_label2idx_path is None:
            print("DEBUG INFO: No Label Encoding needed for this dataset")
            self.label2idx = None
        else:
            label2idx = pd.read_json(s3_label2idx_path, typ='series')
            self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # download image file from s3 location and convert to tensor and normalize.
        # for now, hardcode source of image to aai-540-data s3 bucket
        s3_uri = f"s3://aai-540-data/cct_resized/{row['filename']}"
        with self.fs.open(s3_uri, 'rb') as f:
            image = Image.open(f).convert('RGB')
        image_tensor = self.transform(image)
        
        # get temporal feature vector
        temporal_features = row[['minOfDay_sin', 'minOfDay_cos', 'dayOfYear_sin', 'dayOfYear_cos']].values.astype('float32')
        temporal_tensor = torch.tensor(temporal_features, dtype=torch.float32)

        # if label2idx is present, annotated labels are assumed available, and encoded label output is needed
        if self.label2idx is None:
            label_enc_tensor = torch.tensor(1000, dtype=torch.long)
        else:
            label_enc_tensor = torch.tensor(self.label2idx[row['label']], dtype=torch.long)
        
        return image_tensor, temporal_tensor, label_enc_tensor