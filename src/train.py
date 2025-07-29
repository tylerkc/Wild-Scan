# AAI-590 Group 9
# traininig module using Custom Classifier and Cyclical Temporal Features
# to be updated later
import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import json
from custom_models import AnimalTemporalClassifier
from custom_models import AnimalClassifier
from custom_datasets import S3ImageWithTimeFeatureDataset
from custom_losses import CrossEntropyMarginLoss
import time
import boto3
from typing import Optional
from utils.utils import parse_s3_uri
import torch.optim.lr_scheduler as lr_scheduler


def update_label2idx(existing_label2idx, new_labels):
    label2idx = existing_label2idx.copy() if existing_label2idx else {}
    max_idx = max(label2idx.values()) if label2idx else -1
    for label in new_labels:
        if label not in label2idx:
            max_idx += 1
            label2idx[label] = max_idx
    return label2idx

def train(args):
    
    #print("ARGS:", args)
    #print("ARGS (type):", type(args))
    #print("ARGS epochs: ", args.epochs)
    #print("ARGS train csv: ", args.train_csv)
    #print("ARGS val csv: ", args.val_csv)
    #print("ARGS Label2Idx json: ", args.label2idx_json)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        #print(torch.backends.mps.is_available())  # Should be True
        #print(torch.backends.mps.is_built())      # Should be True
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("DEBUG INFO: DEVICE: ", device)


    # Load or create label2idx mapping
    # DEBUG CHECK ACCESS TO S3 JSON
    #print("DEBUG CHECK: Accessing Label2Idx json file at: ", args.label2idx_json)
    # download label mapping
    # create s3 client from provided session


    session = boto3.Session(profile_name=args.session_profile_name)
    sts = session.client('sts')
    identity = sts.get_caller_identity()
    print("DEBUG INFO: AWS STS Identity: ", identity)
    
    s3_client = session.client('s3')
    s3_client.download_file(
        Bucket="aai-590-tmp2",
        Key="data_split/train_val/label_mapping.json",
        Filename="tmp_label_mapping.json"
    )
    print("DEBUG CHECK: S3 CLIENT DOWNLOAD: (tmp_label_mapping.json) OK")
    with open("tmp_label_mapping.json", "r") as f:
        label2idx = json.load(f)
    #label2idx = pd.read_json(args.label2idx_json, typ='series')
    print("DEBUG CHECK Label2Idx json length: ", len(label2idx))
    #label2idx = load_label2idx(args.label2idx_json)
    #print(f"TRAIN.PY LABEL2IDX checkpoint_(1): {len(label2idx)}")
    #train_df = pd.read_csv(args.train_csv)
    #new_labels = train_df['label'].unique().tolist()
    #print(f"TRAIN.PY LABEL2IDX checkpoint_(2): {new_labels}")
    #label2idx = update_label2idx(label2idx, new_labels)
    #print(f"TRAIN.PY LABEL2IDX checkpoint_(3): {len(label2idx)}")
    # =========DEBUG==================
    #print(json.dumps(label2idx, indent=4))
    
    # Load datasets with consistent label2idx
    train_dataset = S3ImageWithTimeFeatureDataset(args.train_csv, args.label2idx_json, session=session, device = device)

    # =========DEBUG==================
    print(f"DONE: Custom DatasetClass for Training Data{len(train_dataset)}")

    
    val_dataset = S3ImageWithTimeFeatureDataset(args.val_csv, args.label2idx_json, session=session, device = device)

    # =========DEBUG==================
    print(f"DONE: Custom DatasetClass for Validation Data{len(val_dataset)}")
   

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, pin_memory=True)
    #val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    if(args.custom_model == 'AnimalTemporalClassifier'):
        model = AnimalTemporalClassifier(num_classes=len(label2idx)).to(device)
    elif(args.custom_model == 'AnimalClassifier'):
        print("DEBUGDEBUGDEBUG: Base Animal Classifier used for tuning")
        model = AnimalClassifier(num_classes=len(label2idx)).to(device)
    else:
        print("ERROR: invalid Custom Model name specified")
        exit()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    if(args.use_custom_loss):
        criterion = CrossEntropyMarginLoss(reduction = 'mean', margin_lambda = 0.1, margin_type="probs")
        print(f"LOSS FUNCTION: Custom Loss with Margin")

    else:
        criterion = nn.CrossEntropyLoss()
        print(f"LOSS FUNCTION: Cross Entropy")
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }


    for epoch in range(args.epochs):
                  
        # =========DEBUG==================
        print(f"\n===========EPOCH {epoch+1}=================") 
        model.train()
        print("TRAINING...")
        if(args.use_custom_loss):
            criterion.update_params(reduction = 'mean')
            print(f"--Loss Margin Lambda: {criterion.margin_lambda}")
            print(f"--Loss Margin Format: {criterion.margin_type}")
            print(f"--Train Loss Reduction: {criterion.reduction}")
        #print learning rate
        print(f"--Learning Rate: {optimizer.param_groups[0]['lr']}")
        start_train = time.time()
        
        running_loss = 0.0
        running_ce_loss = 0.0
        
        correct_train = 0
        total_train = 0
        batch_idx = 0
        for images, features, labels in train_loader:
            images, features, labels = images.to(device), features.to(device), labels.to(device)
            optimizer.zero_grad()
            if(args.custom_model == 'AnimalTemporalClassifier'):
                outputs = model(images, features)
            else:
                outputs = model(images)
            
            if(args.use_custom_loss):
                loss, ce_loss= criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            if(args.use_custom_loss):
                running_ce_loss += ce_loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            # Print batch number, carriage return to overwrite line
            print(f"--processing batch {batch_idx}/{len(train_loader)}", end='\r', flush=True)
            batch_idx += 1
        print()  # Print a newline after the last batch message
        
        train_loss = running_loss / len(train_dataset)
        train_ce_loss = running_ce_loss / len(train_dataset)
        
        #train_acc = correct_train / total_train
        train_acc = correct_train / len(train_dataset)

        end_train = time.time()
        train_time = end_train - start_train
        print(f"--done! time elapsed: {train_time:.2f} s")
        print(f"--train_acc: {train_acc:.2f}")
        print(f"--train_loss: {train_loss:.2f}, train_ce_loss: {train_ce_loss:.2f}")
        
        scheduler.step()  # Update learning rate scheduler
        
        # Validation
        model.eval()
        print("VALIDATION started....")
        
        start_val = time.time()
        if(args.use_custom_loss):
            criterion.update_params(reduction = 'none')
            print(f"--Val Loss Reduction: {criterion.reduction}")
        correct = 0
        total = 0
        val_loss = 0.0
        val_ce_loss = 0.0
        
        with torch.no_grad():
            for images, features, labels in val_loader:
                images, features, labels = images.to(device), features.to(device), labels.to(device)
                if(args.custom_model == 'AnimalTemporalClassifier'):
                    outputs = model(images, features)
                else:
                    outputs = model(images)
                #outputs = model(images, features)
                #loss = criterion(outputs, labels)
                if(args.use_custom_loss):
                    loss, ce_loss = criterion(outputs, labels)
                else: 
                    loss = criterion(outputs, labels)
                #val_loss += loss.item() * images.size(0)
                #val_ce_loss += ce_loss.item() * images.size(0)
                val_loss += loss.sum().item()
                if(args.use_custom_loss):
                    val_ce_loss += ce_loss.sum().item()
                
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        val_loss = val_loss / len(val_dataset)
        val_ce_loss = val_ce_loss / len(val_dataset)
        
        
        #val_acc = correct / total
        val_acc = correct / len(val_dataset)

        end_val = time.time()
        val_time = end_val - start_val
        print(f"--done! time elapsed: {val_time:.2f} s")
        print(f"--val_acc: {val_acc:.2f}")
        print(f"--val_loss: {val_loss:.2f}, val_ce_loss: {val_ce_loss:.2f}")
        
        # print accuracy and loss values
        #print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        #print(f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        # update learning values
        # Save metrics in history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)


    

    os.makedirs(args.local_model_dir, exist_ok=True)
    #torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    torch.save(model, os.path.join(args.local_model_dir, 'model.pth'))
    
    #save learning history as json
    json.dump(history, open(os.path.join(args.local_model_dir, 'learning_histroy'),'w'))


    #torch.save(model.state_dict(), 'model.pth')
    bucket_name, prefix, _ = parse_s3_uri(args.model_dir)
    s3_client.upload_file(
        Bucket=bucket_name,
        Key=prefix + '/model.pth',
        Filename=os.path.join(args.local_model_dir, 'model.pth')
    )
    
    
    

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '') + '/train-meta.csv')
    parser.add_argument('--val_csv', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '') + '/val-meta.csv')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--label2idx_json', type=str, default=os.environ.get('SM_CHANNEL_LABEL2IDX', '') + '/label_mapping.json')
    parser.add_argument('--custom_model', type=str, default='AnimalTemporalClassifier')
    parser.add_argument('--use_custom_loss', type=bool, default=False)
    parser.add_argument('--local_model_dir', type=str, default='./tmp/model/output')
    parser.add_argument('--session_profile_name', type=str, default="default-sso", help='profile name to initialize a boto3 session for S3 access')
    
    return parser.parse_args(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '') + '/train-meta.csv')
    parser.add_argument('--val_csv', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '') + '/val-meta.csv')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    #parser.add_argument('--label2idx_path', type=str, default=None, help='Path to existing label2idx.json for retraining')
    #parser.add_argument('--label2idx_path', type=str, default=os.path.join(os.environ.get('SM_CHANNEL_LABEL2IDX', ''), '/label_mapping.json'))
    parser.add_argument('--label2idx_json', type=str, default=os.environ.get('SM_CHANNEL_LABEL2IDX', '') + '/label_mapping.json')
    
    parser.add_argument('--custom_model', type=str, default='AnimalTemporalClassifier')
    parser.add_argument('--use_custom_loss', type=bool, default=False)
    parser.add_argument('--local_model_dir', type=str, default='./tmp/model/output')
    parser.add_argument('--session_profile_name', type=str, default="default-sso", help='profile name to initialize a boto3 session for S3 access')
    args = parser.parse_args()
    train(args)