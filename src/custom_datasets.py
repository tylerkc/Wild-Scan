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

import logging
# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
)
logger = logging.getLogger(__name__)


# add s3 session credentials to this script
class S3ImageWithTimeFeatureDataset(Dataset):
    
    def __init__(self, 
                 csv_path = None, label2idx_path = None, images_path: str = './images', session = None, s3_access_profile = None, 
                 meta_df: pd.DataFrame=None, label2idx: dict=None):

        # load meta data from either csv_path or meta_df
        self.df = None
        if csv_path is not None:
            if csv_path.startswith("s3://"):
                # parse the s3 uri
                bucket_name, prefix, filename = parse_s3_uri(csv_path)
                logger.debug(f"input csv s3: {bucket_name}, {prefix}, {filename}")
                # create a local temporary directory to download files
                if session is None:
                    # create a boto3 session with default profile
                    session = boto3.Session(profile_name=s3_access_profile)
                s3_client = session.client('s3')

                # download merged inferences and label mapping
                s3_client.download_file(
                    Bucket=bucket_name,
                    Key = f"{prefix}/{filename}",
                    Filename = filename
                )
                self.df = pd.read_csv(filename)
                
            else:
                # assume local file path
                logger.debug(f"input local csv: {csv_path}")
                self.df = pd.read_csv(csv_path)
                
        elif meta_df is not None:
            # if meta_df is provided, use it directly
            logger.debug("Using provided meta_df DataFrame instead")
            self.df = meta_df
            
        else:
            logger.error("No CSV path or meta_df provided. Cannot initialize dataset.")
            sys.exit(1)

        
        
        logger.info(f"Dataset shape: {self.df.shape}")
        self.size = len(self.df)
        # transform an input Image into a Tensor then normalize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # save images path
        self.images_path = images_path

        # use s3fs is self.images_path is an s3 bucket, otherwise use local directory
        self.use_local_dir = not self.images_path.startswith("s3://")
        if(self.use_local_dir):
            logger.debug("no need to initialize s3fs since using local directory for dataset")
        else:
            logger.debug(f"using s3fs for dataset access using s3 sso profile name {s3_access_profile}")
            credentials = session.get_credentials().get_frozen_credentials()
            self.fs = s3fs.S3FileSystem(
                key=credentials.access_key,
                secret=credentials.secret_key,
                token=credentials.token,
                client_kwargs={
                    'region_name': session.region_name
                }
            
            )
            logger.info("s3fs.Filesystem initialization successful")

       

        # load label2idx mapping from the path, check if it's from s3 or local
        if label2idx_path is not None:
            if label2idx_path.startswith("s3://"):
                # parse the s3 uri
                bucket_name, prefix, filename = parse_s3_uri(label2idx_path)
                logger.info(f"downloading label_mnapping.json (label2idx) from s3: {bucket_name}, {prefix}, {filename}")
                # create a local temporary directory to download files
                if session is None:
                    # create a boto3 session with default profile
                    session = boto3.Session(profile_name=s3_access_profile)
                s3_client = session.client('s3')

                # download label2idx mapping
                s3_client.download_file(
                    Bucket=bucket_name,
                    Key = f"{prefix}/{filename}",
                    Filename = filename
                )
                with open(filename, "r") as f:
                    label2idx = json.load(f)
                self.label2idx = label2idx
            else:
                # assume local file path
                logger.info(f"Loading label2idx mapping from local file: {label2idx_path}")
                with open(label2idx_path, "r") as f:
                    label2idx = json.load(f)
                self.label2idx = label2idx
        elif label2idx is not None:
            # if label2idx is provided as a dict, use it directly
            logger.info("loading label2idx mapping from provided dict")
        
            self.label2idx = label2idx
        else:
            # if no label2idx is provided, assume no label encoding is needed
            logger.info("No Label Encoding needed for this dataset")
            self.label2idx = None

        #self.device = torch.device(device)
        logger.info(f"Dataset initialized with {self.size} samples.")



    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # download image file from s3 location and convert to tensor and normalize.
        # for now, hardcode source of image to aai-540-data s3 bucket
        image_path = self.images_path + row['filename']
        if self.use_local_dir:
            #image_path = '../dataset/cct_resized/' + row['filename']
            
            # open the image file directly from local directory
            #print(f"--DEBUG INFO: Opening image from local directory: {image_path}")
            image = Image.open(image_path).convert('RGB')
        else:
            #image_path = f"s3://aai-540-data/cct_resized/{row['filename']}"
            # use s3fs to open the file
            #print(f"--DEBUG INFO: Opening image from S3: {image_path}")
            with self.fs.open(image_path, 'rb') as f:
                image = Image.open(f).convert('RGB')
            # image_tensor = self.transform(image)

        #s3_uri = f"s3://aai-540-data/cct_resized/{row['filename']}"
        #with self.fs.open(s3_uri, 'rb') as f:
        #    image = Image.open(f).convert('RGB')
        image_tensor = self.transform(image)
        
        # get temporal feature vector
        temporal_features = row[['minOfDay_sin', 'minOfDay_cos', 'dayOfYear_sin', 'dayOfYear_cos']].values.astype('float32')
        temporal_tensor = torch.tensor(temporal_features, dtype=torch.float32)

        # if label2idx is present, annotated labels are assumed available, and integer encoded label output is needed
        if self.label2idx is None:
            label_enc_tensor = torch.tensor(1000, dtype=torch.long)
        else:
            label_enc_tensor = torch.tensor(self.label2idx[row['label']], dtype=torch.long)
        
        return image_tensor, temporal_tensor, label_enc_tensor
        #return image_tensor.to(self.device), temporal_tensor.to(self.device), label_enc_tensor.to(self.device)



