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


def update_label2idx(existing_label2idx, new_labels):
    label2idx = existing_label2idx.copy() if existing_label2idx else {}
    max_idx = max(label2idx.values()) if label2idx else -1
    for label in new_labels:
        if label not in label2idx:
            max_idx += 1
            label2idx[label] = max_idx
    return label2idx

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE: ", device)
    print("ARGS train csv: ", args.train_csv)
    print("ARGS val csv: ", args.val_csv)
    print("ARGS Label2Idx json: ", args.label2idx_json)

    # Load or create label2idx mapping
    label2idx = pd.read_json(args.label2idx_json, typ='series')
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
    train_dataset = S3ImageWithTimeFeatureDataset(args.train_csv, args.label2idx_json)

    # =========DEBUG==================
    print(f"DONE: Custom DatasetClass for Training Data{len(train_dataset)}")

    
    val_dataset = S3ImageWithTimeFeatureDataset(args.val_csv, args.label2idx_json)

    # =========DEBUG==================
    print(f"DONE: Custom DatasetClass for Validation Data{len(val_dataset)}")
   

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    #val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if(args.custom_model == 'AnimalTemporalClassifier'):
        model = AnimalTemporalClassifier(num_classes=len(label2idx)).to(device)
    elif(args.custom_model == 'AnimalClassifier'):
        print("DEBUGDEBUGDEBUG: Base Animal Classifier used for tuning")
        model = AnimalClassifier(num_classes=len(label2idx)).to(device)
    else:
        print("ERROR: invalid Custom Model name specified")
        exit()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #criterion = nn.CrossEntropyLoss()
    criterion = CrossEntropyMarginLoss(reduction = 'mean', margin_lambda = 0.1, margin_type="probs")
    

    for epoch in range(args.epochs):
        criterion.update_params(reduction = 'mean')
        # =========DEBUG==================
        print(f"\n===========EPOCH {epoch+1}=================")
        print(f"HYPER_PARAMETERS:")
        print(f"--Loss Margin Lambda: {criterion.margin_lambda}")
        print(f"--Loss Margin Format: {criterion.margin_type}")
        
        
        
        model.train()
        print("TRAINING...")
        print(f"--Train Loss Reduction: {criterion.reduction}")
        start_train = time.time()
        
        running_loss = 0.0
        running_ce_loss = 0.0
        
        correct_train = 0
        total_train = 0
        
        for images, features, labels in train_loader:
            images, features, labels = images.to(device), features.to(device), labels.to(device)
            optimizer.zero_grad()
            if(args.custom_model == 'AnimalTemporalClassifier'):
                outputs = model(images, features)
            else:
                outputs = model(images)
            loss, ce_loss= criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            running_ce_loss += ce_loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
        
        train_loss = running_loss / len(train_dataset)
        train_ce_loss = running_ce_loss / len(train_dataset)
        
        #train_acc = correct_train / total_train
        train_acc = correct_train / len(train_dataset)

        end_train = time.time()
        train_time = end_train - start_train
        print(f"--done! time elapsed: {train_time:.2f} s")
        print(f"--train_acc: {train_acc:.2f}")
        print(f"--train_loss: {train_loss:.2f}, train_ce_loss: {train_ce_loss:.2f}")
        

        
        # Validation
        model.eval()
        print("VALIDATION started....")
        
        start_val = time.time()

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
                
                loss, ce_loss = criterion(outputs, labels)
                #val_loss += loss.item() * images.size(0)
                #val_ce_loss += ce_loss.item() * images.size(0)
                val_loss += loss.sum().item()
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

    # Save model and updated label2idx mapping
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    #with open(os.path.join(args.model_dir, 'label2idx.json'), 'w') as f:
        #json.dump(label2idx, f)

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
    args = parser.parse_args()
    train(args)