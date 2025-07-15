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

def model_fn(model_dir):
    model = torch.load(os.path.join(model_dir, 'model.pth'), map_location='cpu')
    model.eval()
    with open(os.path.join(model_dir, 'label2idx.json')) as f:
        label_map = json.load(f)
    return model, label_map

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

def predict_fn(input_data, model_and_map):
    model, label_map =
    with torch.no_grad():
        outputs = model(input_data)
        _, predicted = torch.max(outputs, 1)
        idx_to_label = {int(v): k for k, v in label_map.items()}
        return idx_to_label[predicted.item()]

def output_fn(prediction, accept):
    return str(prediction), accept