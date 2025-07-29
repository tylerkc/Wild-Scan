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
import os
from utils.utils import parse_s3_uri
import boto3
import json

# add s3 session credentials to this script
class S3ImageWithTimeFeatureDataset(Dataset):
    
    def __init__(self, s3_csv_path, s3_label2idx_path = None, session = None, s3_access_profile = None, device = "mps"):

        # load meta data csv to process into pandas df
        #self.df = pd.read_csv(s3_csv_path)
        # parse s3 uri to get bucket name and prefix    
        bucket_name, prefix, filename = parse_s3_uri(s3_csv_path)
        print(f"DEBUG input csv s3: {bucket_name}, {prefix}, {filename}")
        # create a local temporary directory to download files
        
        if session is None:
            # create a boto3 session with default profile
            session = boto3.Session(profile_name=s3_access_profile)
        s3_client = session.client('s3')

        # download merged inferences and label mapping
        s3_client.download_file(
            Bucket=bucket_name,
            #Key="data_split/train_val/validation2/evaluation/merged_inferences.csv",
            Key = f"{prefix}/{filename}",
            #Filename=f"{local_tmp_dir}/merged_inferences.csv"
            Filename = filename
        )
        self.df = pd.read_csv(filename)

        print(f"DEBUG INFO: self.df.shape = {self.df.shape}")

        # transform an input Image into a Tensor then normalize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # initialize s3fs access 
        #self.fs = s3fs.S3FileSystem()
        credentials = session.get_credentials().get_frozen_credentials()
        self.fs = s3fs.S3FileSystem(
            key=credentials.access_key,
            secret=credentials.secret_key,
            token=credentials.token,
            client_kwargs={
                'region_name': session.region_name
            }
        )

        print(f"DEBUG INFO: self.fs OK")

        # if label2idx path is given, it is assumed that label encoding is required
        if s3_label2idx_path is None:
            print("DEBUG INFO: No Label Encoding needed for this dataset")
            self.label2idx = None
        else:
            #label2idx = pd.read_json(s3_label2idx_path, typ='series')
            #self.label2idx = label2idx

            bucket_name, prefix, filename = parse_s3_uri(s3_label2idx_path)
            print(f"DEBUG label mapping json s3: {bucket_name}, {prefix}, {filename}")
            s3_client.download_file(
                Bucket=bucket_name,
                Key = f"{prefix}/{filename}",
                Filename = filename
            )
            #print("DEBUG CHECK: S3 CLIENT DOWNLOAD: (tmp_label_mapping.json) OK")
            with open(filename, "r") as f:
                label2idx = json.load(f)
            self.label2idx = label2idx
            print(f"DEBUG INFO: self.label2idx = {self.label2idx}")
        print(f"DEBUG INFO: self.label2idx OK")

        self.device = torch.device(device)
        print(f"DEBUG INFO: Dataset initialized with device {self.device}")



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
        #return image_tensor.to(self.device), temporal_tensor.to(self.device), label_enc_tensor.to(self.device)