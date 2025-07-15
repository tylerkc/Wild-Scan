# evaluate.py
# gets executed during 
import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
import logging
from datetime import datetime
import os

def create_prediction_df(input_dir: str) -> pd.DataFrame:
    """
    combines transform .out files into one dataframe. 
    output df contains three columns:
        - 'prediction': list of pred probabilities    
    """
    # Verify input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # iterate through each prediction file
    records = []

    record_count = 0
    # Process each .out file in directory
    for filename in os.listdir(input_dir):
        if not filename.endswith('.out'):
            continue
            
        file_path = os.path.join(input_dir, filename)
        record_count += 1
        try:
            pred_probs_df = pd.read_json(file_path)
            predictions = pred_probs_df['prediction'].to_list()

            # extract base filename for merging with ground truth later
            base_name = os.path.splitext(filename)[0] 
            records.append({
                'filename': base_name,
                'prediction': predictions,
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
           
    records_df = pd.DataFrame(records)
    #print(records_df.head())
    print(f"pred probs df shape:{records_df.shape}")
    
    return records_df


def main():
    # input arguments
    PREDICTIONS_DIR = '/opt/ml/processing/input_predictions'  
    TRUE_LABELS_DIR = '/opt/ml/processing/true_labels'
    LABEL_MAPPING_PATH = '/opt/ml/processing/label_mapping/label_mapping.json'
    OUTPUT_DIR = '/opt/ml/processing/output'
    
    ##### LOGGING ########
    # Set up logging to output to stdout
    logging.basicConfig(level=logging.INFO)
    print(f"[{datetime.now()}] Script has started.")
    logging.info(f"Script has started at {datetime.now()}.")
    number_of_files = sum(1 for item in os.listdir(PREDICTIONS_DIR) if os.path.isfile(os.path.join(PREDICTIONS_DIR, item)))
    print(f"Number of files in {PREDICTIONS_DIR}: {number_of_files}")
    ##### LOGGING ########
    
    # STEP 1. generate dataframe with 'filename' column and 'prediction' column containing pred_probs
    preds_df = create_prediction_df(PREDICTIONS_DIR)
    print(f" (step 1) pred probs df shape: {preds_df.shape}")

    # STEP 2. Load CSV file for the month listing new files with ground truth labels
    csv_files = [f for f in os.listdir(TRUE_LABELS_DIR) if f.endswith('.csv')]
    TRUE_LABELS_PATH = os.path.join(TRUE_LABELS_DIR, csv_files[0])
    labels_df = pd.read_csv(TRUE_LABELS_PATH)
    print(f" (step 2) True Labels data shape: {labels_df.shape}")
    #print(labels_df.head())

    # STEP 3. merge pred proba df with true labels df on 'filename'
    merged_df = pd.merge(preds_df, labels_df, on='filename', how='inner')
    print(f" (step3) Merged data shape : {merged_df.shape}")
    #print(merged_df.head())
    
    # STEP 4. Convert Highest Pred Prob to Actual Label string and add columns to merged #############
    with open(LABEL_MAPPING_PATH, 'r') as f:
        label_to_index = json.load(f)
    index_to_label = {v: k for k, v in label_to_index.items()}
    merged_df['pred_label_enc'] = merged_df['prediction'].apply(lambda x: x.index(max(x)))
    merged_df['pred_label'] = merged_df['pred_label_enc'].map(index_to_label)
    print(f" (step4) Merged data shape with preds: {merged_df.shape}")
    #print(merged_df.head())
    
    # STEP 5. CALCULATE METRICS
    print(f" (step5) METRICS CALCULATION")

    ## 5.1 Class Restriced Metrics (accuracy, f1-score, classification_report) - Evaluate only on labels available during training
    ## avoids new class contamination 
    class_restrict_df = merged_df[merged_df['label'].isin(label_to_index.keys())] # known classes
    print(f" (step5.1) Class-Restricted metrics - Evaluate only on labels available during training")
    print(f" (step5.1) class_restriced df shape : {class_restrict_df.shape}")
    #print(class_restricted_df.head())
    class_restrict_accuracy = accuracy_score(class_restrict_df['label'], class_restrict_df['pred_label'])
    class_restrict_f1 = f1_score(class_restrict_df['label'], class_restrict_df['pred_label'], average = 'weighted')
    class_restrict_report = classification_report(class_restrict_df['label'], class_restrict_df['pred_label'])
    print(class_restrict_report)

    ## 5.2 Novelty Ratio Metric: portion of the new dataset that is unfamiliar to the model given a confidence threshold
    print(f" (step5.2) Novelty Ratio Metric: portion of the new dataset that is unfamiliar to the model given a confidence threshold=0.7")
    NOVELTY_THRESHOLD = 0.7
    merged_df['max_pred_prob'] = merged_df['prediction'].apply(max)
    novel_samples_cnt = (merged_df['max_pred_prob'] < NOVELTY_THRESHOLD).sum()
    novelty_ratio = novel_samples_cnt / len(merged_df)
    print(f" (step5.2) novelty ratio : {novelty_ratio}")
    
    # STEP 6. Generate OUTPUT FILES
    ## 6.1 save restricted classification report as csv
    class_restrict_report_dict = classification_report(class_restrict_df['label'], class_restrict_df['pred_label'], output_dict=True)
    report_df = pd.DataFrame(class_restrict_report_dict).transpose()
    report_df.to_csv(f'{OUTPUT_DIR}/restricted_class_report.csv')

    ## 6.2 save other metrics in json file
    ### Save the full report as a DataFrame for later access
    report_dict = {
        "multiclass_classification_metrics": {
            "accuracy": {"value": class_restrict_accuracy, "standard_deviation": "NaN"},
            "f1-weighted": {"value": class_restrict_f1, "standard_deviation": "NaN"},
            "novelty_ratio": {"value": novelty_ratio, "standard_deviation": "NaN"}
        }
    }
    with open(f'{OUTPUT_DIR}/multiclass_metrics.json', "w") as f:
        json.dump(report_dict, f)

    ## 6.3 save the final merged df for later access
    merged_df.to_csv(f'{OUTPUT_DIR}/merged_inferences.csv')

    print("Done executing script")
    
if __name__ == "__main__":
    main()