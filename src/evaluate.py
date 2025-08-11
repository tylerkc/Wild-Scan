# evaluate.py
# evaluation pipe script
# AAI-590 Group 9
# does the following steps:
# 1. loads recent model to device 
# 2. 

import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
import logging
from datetime import datetime
import os
import argparse
import inference as inference
from custom_datasets import S3ImageWithTimeFeatureDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import confidence_estimator as ConfidenceEstimator
import torch
import boto3





def make_decision(calibrated_confidence, thresh_high=0.87, thresh_med=0.65, thresh_low=0.5):
    if calibrated_confidence >= thresh_high:
        return "ACCEPT", "High confidence prediction"
    elif calibrated_confidence >= thresh_med:
        return "ACCEPT_WITH_FLAG", "Medium confidence - flag for review"
    elif calibrated_confidence >= thresh_low:
        return "REVIEW", "Low confidence - human review recommended"
    else:
        return "REJECT", "Very low confidence - reject prediction"
    
def main(args):
    # Step 0. Set up directories and paths

    # ==============================
    # STEP 1. Load model
    print("STEP1: Load model")
    model = inference.load_model(args.model_path)
    
    # ==============================
    # STEP 2. Setup dataset 
    print("\nSTEP2: Setup dataset")
    # load label map
    label2idx = pd.read_json(args.label2idx_json, typ="series").to_dict()
    idx2label = {v: k for k, v in label2idx.items()}
    NUM_CLASSES = len(label2idx)

    # s3 dataset to evaluate
    #session = boto3.Session(profile_name=args.session_profile_name)
    session = boto3.Session(profile_name=args.session_profile_name)
    sts = session.client('sts')
    
    identity = sts.get_caller_identity()
    print("--DEBUG INFO: session profile name: ", args.session_profile_name)
    print("--DEBUG INFO: AWS STS Identity: ", identity)
    new_dataset = []
    new_dataset = S3ImageWithTimeFeatureDataset(args.new_data_csv, session = session)
    print(f"--Number of images: {len(new_dataset)}")
    
    # data loader for batch processing
    new_dataset_loader = []
    new_dataset_loader = DataLoader(new_dataset, batch_size=32, shuffle=False, num_workers=8)
    print(f"--Number of batches: {len(new_dataset_loader)}")

    
    # ==============================
    # STEP 3. Run inference on new dataset
    print("\nSTEP3: Run inference on new dataset")
    # batch transform validation set for evaluation
    # convert args.use_tempora_features to boolean
    
    print("--DEBUG INFO: use_temporal_features: ", args.use_temporal_features)
    pred_logits, pred_labels = inference.batch_transform(
        model, new_dataset_loader, idx2label, bool(args.use_temporal_features)
    )

    # STEP 3.1. save predictions to output directory
    print("STEP3.1: Save predictions to output directory")
    # create output directory if it does not exist
    OUTPUT_DIR = args.output_dir
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # define output file paths
    PRED_PROBS_FILE = os.path.join(OUTPUT_DIR,'pred_probs.csv')
    PRED_LOGITS_FILE = os.path.join(OUTPUT_DIR, 'pred_logits.csv')
    PRED_LABELS_FILE = os.path.join(OUTPUT_DIR, 'pred_labels.csv')
    # save predictions (logits, probabilities, labels) to specific locations
    pred_probs = F.softmax(pred_logits, dim=1).numpy()
    pred_logits_df = pd.DataFrame(pred_logits.numpy(), columns = label2idx.keys())
    pred_logits_df.to_csv(PRED_LOGITS_FILE, index = False)
    pred_probs_df = pd.DataFrame(pred_probs, columns = label2idx.keys())
    pred_probs_df.to_csv(PRED_PROBS_FILE, index = False)
    # save predicted labels to file
    pred_labels_df = pd.DataFrame(pred_labels, columns=['predicted_label'])
    pred_labels_df.to_csv(PRED_LABELS_FILE, index=False)

    # ===================================
    # STEP 4. CONFIDENCE ESTIMATION (TvA) 
    # TopVersusAll (TvA) calibration does not change the model's original predictions (which was based on softmax probabilities)
    # This step is used to label uncertain predictions as "novel" or "unfamiliar" to the model 
    print("\nSTEP4: Confidence Estimation (TvA Calibration)")
    # step 4.1. initialize confidence estimator
    conf_estimator = []
    conf_estimator = ConfidenceEstimator.ConfidenceEstimator(NUM_CLASSES)

    # step 4.2 calibrate confidence estimator on calibration set
    # load calibration data
    print("STEP4.2: Calibrate confidence estimator on calibration set")
    cal_pred_logits_df = pd.read_csv(args.cal_pred_logits_csv)
    print("--DEBUG INFO: calibration pred logits loaded (shape): ", cal_pred_logits_df.shape)
    cal_true_labels_df = pd.read_csv(args.cal_true_labels_csv)
    cal_true_labels = cal_true_labels_df['true_labels'].tolist()
    # print class distribution of true labels of calibration set
    print("--DEBUG INFO: calibration true labels loaded (shape): ", cal_true_labels_df.shape)
    print("--DEBUG INFO: calibration true labels distribution: ", cal_true_labels_df['true_labels'].value_counts())
    cal_true_labels_idx = [label2idx[label] for label in cal_true_labels]
    # convert to tensor
    cal_pred_logits = torch.tensor(cal_pred_logits_df.values, dtype=torch.float32)
    cal_true_labels_tensor = torch.tensor(cal_true_labels_idx, dtype=torch.long)
    # calibrate
    conf_estimator.calibrate(cal_pred_logits, cal_true_labels_tensor)

    # step 4.3 compute confidence scores on new dataset
    conf_estimator.update_statistics(pred_logits)
    conf_report = conf_estimator.get_confidence_report()
    # save confidence estimates.csv
    print("STEP4.3: Compute confidence scores on new dataset")
    original_confidences = pred_probs_df.max(axis=1)
    tva_confidences = conf_estimator.global_stats['calibrated_confidence']
    tva_confidences_df = pd.DataFrame(tva_confidences, columns=['calibrated_confidence'])
    tva_confidences_df.to_csv(os.path.join(OUTPUT_DIR, 'tva_confidences.csv'), index=False)

    # create a df where row is a sample and columns are: original confidence, tva confidence, true_label, predicted label
    confidences_df = pd.DataFrame({
        'original_confidence': original_confidences,
        'tva_confidence': tva_confidences,
        'pred_label': pred_labels,
    })
    # apply decisions based on tva confidence and thresholds
    confidences_df['decision'], confidences_df['decision_reason'] = zip(*confidences_df['tva_confidence'].apply(make_decision, args = (args.conf_thresh_high, args.conf_thresh_med, args.conf_thresh_low)))
    # save confidences_df to output directory
    confidences_df.to_csv(os.path.join(OUTPUT_DIR, 'combined_confidences.csv'), index=False)

    per_class_df = pd.DataFrame({})
    for i in range(0,NUM_CLASSES):
        per_class_df.loc[i, 'pred_label']= idx2label[i]
        per_class_df.loc[i, 'pred_count'] = conf_report['per_class'][f'class_{i}']['calibrated_confidence']['count'] 
        per_class_df.loc[i, 'orig conf (mean)'] = conf_report['per_class'][f'class_{i}']['original_confidence']['mean']
        per_class_df.loc[i, 'cal conf (mean)'] = conf_report['per_class'][f'class_{i}']['calibrated_confidence']['mean']

    # ===================================
    # STEP 5. Display some results 
    # show decision counts
    print("\nSTEP5: Unsupevised Reports")
    decision_counts = confidences_df['decision'].value_counts()
    print("\n5.1 Uncertainty Counts (After TvA Calibration):")
    print(decision_counts)

    print("\n5.2 Global Confidence Report:")
    print(f"-- Original Confidence: {conf_report['global']['original_confidence']['mean']}")
    print(f"-- TvA Confidence: {conf_report['global']['calibrated_confidence']['mean']}")

    print("\n5.3 Per Class Confidence Report:")
    display(per_class_df)

    

    print("Done executing script")

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Evaluate model predictions against true labels.')
    
    # model params
    parser.add_argument('--model_name', type=str, default='ResNet18_Finetuned')
    parser.add_argument('--model_path', type=str, default = './model.pth')
    parser.add_argument('--use_temporal_features', type=bool, default=False)

    # input data params
    parser.add_argument('--new_data_csv', type=str, default='./data_split/production/test-meta.csv')
    parser.add_argument('--label2idx_json', type=str, default='./data_split/train_val/label_mapping.json')

    # calibration params
    parser.add_argument('--cal_pred_logits_csv', type=str, default='./evaluation/model3/pred_logits.csv')
    parser.add_argument('--cal_true_labels_csv', type=str, default='./evaluation/model3/true_labels.csv')

    # confidence params
    parser.add_argument('--conf_thresh_high', type=float, default=0.87, help='confidence threshold for decision making')
    parser.add_argument('--conf_thresh_med', type=float, default=0.6, help='confidence threshold for decision making')
    parser.add_argument('--conf_thresh_low', type=float, default=0.4, help='confidence threshold for decision making')
    
    # output dir for preds and evals
    parser.add_argument('--output_dir', type=str, default=os.environ.get('./evaluation/output'))

    # s3 session profile
    parser.add_argument('--session_profile_name', type=str, default="default-sso", help='profile name to initialize a boto3 session for S3 access')

    # for debug
    #parser.add_argument('--debug_skip_inference', type=bool, default=False)

    

    return parser.parse_args(args)
    #return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    
    main(args)
    

