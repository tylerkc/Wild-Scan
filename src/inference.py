# AAI-590 Group 9
# Inference Script
# Required modules to make an inference on any SINGLE NEW IMAGE
# to be updated later

# inference.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import json
import os
import boto3
import time
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
)
logger = logging.getLogger(__name__)

def load_model(model_path):
    
    # use gpu by default if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Available Device for Model Mapping: {device}")
    
    # load model, make sure it was saved via torch.save(model, path) and not just model.state_dict()
    model = []
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully for inference from: {model_path}")

    return model



def input_fn(request_body, request_content_type):
    # Each request_body is a line from the CSV (e.g., "cat.jpg")
    if request_content_type == 'text/csv':
        filename = request_body.decode('utf-8').strip()
        #bucket = os.environ.get('IMAGE_BUCKET')  # Set in environment
        bucket = "aai-540-data"
        #prefix = os.environ.get('IMAGE_PREFIX', '')  # Optional subfolder
        prefix = "cct_resized"
        s3 = boto3.client('s3')
        #s3_uri = f"s3://aai-540-data/cct_resized/{row['filename']}"
        key = os.path.join(prefix, filename) if prefix else filename
        img_bytes = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        transform = transforms.Compose([
            #transforms.Resize(224),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    else:
        raise Exception(f"Unsupported content                       type: {request_content_type}")

#def predict_fn(input_data, model_and_map):
#    model, label_map =
#    with torch.no_grad():
#        outputs = model(input_data)
#        _, predicted = torch.max(outputs, 1)
#        idx_to_label = {int(v): k for k, v in label_map.items()}
#        return idx_to_label[predicted.item()]

#def output_fn(prediction, accept):
#    return str(prediction), accept



def batch_transform(model, new_dataset_loader, label_mapping: dict = None, use_temporal_features=False):
    # use gpu by default if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.debug(f"Available Device for Batch Transform: {device}")
    
    pred_labels = []
    pred_logits = []
    batch_id = 0

    start_time = time.time()
    model.eval()
    model.to(device)

    idx2label = {v: k for k, v in label_mapping.items()}

    with torch.no_grad():
        for images_batch, features_batch, scalars_batch in new_dataset_loader:
            batch_start_time = time.time()
            images, features = images_batch.to(device), features_batch.to(device)
            if use_temporal_features:
                outputs = model(images, features)
            else:
                outputs = model(images) 
            
            _, predicted = torch.max(outputs, 1)
            pred_labels_batch = []
            pred_labels_batch = [idx2label[int(idx)] for idx in predicted]

            batch_end_time = time.time()
            elapsed_batch_time = batch_end_time - batch_start_time
            running_elapsed_time = batch_end_time - start_time
            print(f"--processed batch {batch_id}/{len(new_dataset_loader)} elapsed time: batch [{elapsed_batch_time} s]  total [{running_elapsed_time} s]", end='\r', flush=True)
            batch_id += 1
            pred_labels.extend(pred_labels_batch)
            pred_logits.extend(outputs)

    pred_logits = torch.stack(pred_logits).cpu()
    
    return pred_logits, pred_labels

