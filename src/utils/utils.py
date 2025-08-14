def parse_s3_uri(s3_uri):
    from urllib.parse import urlparse
    from pathlib import PurePosixPath
    import os
    
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    prefix_folder = '/'.join(key.split('/')[:-1]) if '/' in key else ''
    filename = PurePosixPath(key).name
    return bucket, prefix_folder, filename


def generate_manifest_file(s3_images_loc, s3_input_csv):
    # assumes s3 bucket access is available
    # creates a manifest file based on the input csv, and uploads it to the same s3 directory as the input_csv.
    # s3_images_loc is the common directory where all the images files are
    import os
    import pandas as pd
    import boto3
    from botocore.exceptions import ClientError
    import json
    
    df = pd.read_csv(s3_input_csv)
    manifest = [{"prefix": s3_images_loc}] + df['filename'].to_list()
    
    # parse s3uri of input csv
    bucket_name, prefix, filename = parse_s3_uri(s3_input_csv)

    # create manifest output name using the same filename as input
    output_filename = os.path.splitext(filename)[0] + ".manifest"
    
    s3_key = f"{prefix}/{output_filename}"
    
    # Write the manifest to a JSON file (for local, delete later)
    with open(output_filename, "w") as f:
        json.dump(manifest, f, indent=2)

    
    # upload manifest to s3 location same as input csv
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(output_filename, bucket_name, s3_key)
        print(f"File uploaded to s3://{bucket_name}/{s3_key}")
    except ClientError as e:
        print(f"Error uploading file: {e}")

    return

def draw_bbox(image_path, bbox, outline ='red'):
    from PIL import Image, ImageDraw
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    # Draw rectangle using bbox coordinates
    bbox = [ bbox[0]/2, bbox[1]/2, (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2,  ]  # Convert to [x1, y1, x2, y2] format
    #display(bbox)
    draw.rectangle(bbox, outline=outline, width=2)
    return image

def display_image_with_info_compact(tmp_df, id, eccv18_imgs_path):
    import os
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    
    
    image_id = tmp_df.iloc[id]['image_id']
    image_path = os.path.join(eccv18_imgs_path, image_id + ".jpg")
    # check if image_path exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    """Compact version for multiple sample display"""
    fig = plt.figure(figsize=(8, 4.5))  # Reduced from (14, 8)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1])
    
    # Image subplot
    ax_img = plt.subplot(gs[0])
    
    bbox = eval(tmp_df.iloc[id]['bbox'])
    
    image = draw_bbox(image_path, bbox)
    ax_img.imshow(image)
    ax_img.axis('off')
    ax_img.set_title(f"Sample ID: {image_id}", fontsize=9, pad=5)
    
    # Info panel
    ax_text = plt.subplot(gs[1])
    ax_text.axis('off')
    ax_text.set_facecolor('#f8f9fa')
    
    # Extract data
    pred_label = tmp_df.iloc[id]['pred_label']
    true_label = tmp_df.iloc[id]['label']
    orig_conf = tmp_df.iloc[id]['orig_confidence']
    cal_conf = tmp_df.iloc[id]['calibrated_confidence']
    uncertainty = tmp_df.iloc[id]['uncertainty_mask']
    
    # Scaled-down text elements
    ax_text.text(0.5, 0.88, "PREDICTION", 
                transform=ax_text.transAxes, ha='center',
                fontsize=8, fontweight='bold', color='#2c3e50')
    
    ax_text.text(0.5, 0.78, str(pred_label),
                transform=ax_text.transAxes, ha='center',
                fontsize=12, fontweight='bold', color='#34495e',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Uncertainty flag
    flag_color = '#e74c3c' if uncertainty else '#27ae60'
    flag_text = "⚠️ REVIEW" if uncertainty else "✓ CONFIDENT"
    
    ax_text.text(0.5, 0.62, flag_text,
                transform=ax_text.transAxes, ha='center',
                fontsize=9, fontweight='bold', color=flag_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=flag_color, alpha=0.1, edgecolor=flag_color))
    
    # Confidence scores
    ax_text.text(0.5, 0.45, "Confidence",
                transform=ax_text.transAxes, ha='center',
                fontsize=7, fontweight='bold', color='#7f8c8d')
    
    ax_text.text(0.5, 0.38, f"Orig: {orig_conf:.3f}",
                transform=ax_text.transAxes, ha='center',
                fontsize=7, color='#95a5a6')
    
    ax_text.text(0.5, 0.32, f"Cal: {cal_conf:.3f}",
                transform=ax_text.transAxes, ha='center',
                fontsize=7, color='#95a5a6')
    
    # Ground truth
    ax_text.text(0.5, 0.12, "Ground Truth",
                transform=ax_text.transAxes, ha='center',
                fontsize=7, style='italic', color='#bdc3c7')
    
    ax_text.text(0.5, 0.08, f"({true_label})",
                transform=ax_text.transAxes, ha='center',
                fontsize=7, style='italic', color='#bdc3c7', alpha=0.9)
    
    plt.tight_layout()
    return fig

def display_image_with_info(tmp_df, id, eccv18_imgs_path):
    import os
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1])
    
    # Image subplot
    ax_img = plt.subplot(gs[0])
    image_id = tmp_df.iloc[id]['image_id']
    image_path = os.path.join(eccv18_imgs_path, image_id + ".jpg")
     # check if image_path exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    bbox = eval(tmp_df.iloc[id]['bbox'])
    
    image = draw_bbox(image_path, bbox)
    ax_img.imshow(image)
    ax_img.axis('off')
    ax_img.set_title(f"Sample ID: {image_id}", fontsize=12, pad=10)
    
    # Info panel
    ax_text = plt.subplot(gs[1])
    ax_text.axis('off')
    ax_text.set_facecolor('#f8f9fa')  # Light background
    
    # Extract data
    pred_label = tmp_df.iloc[id]['pred_label']
    true_label = tmp_df.iloc[id]['label']
    orig_conf = tmp_df.iloc[id]['orig_confidence']
    cal_conf = tmp_df.iloc[id]['calibrated_confidence']
    uncertainty = tmp_df.iloc[id]['uncertainty_mask']
    
    # 1. PREDICTION (Most prominent)
    ax_text.text(0.5, 0.85, "PREDICTION", 
                transform=ax_text.transAxes, ha='center',
                fontsize=12, fontweight='bold', color='#2c3e50')
    
    ax_text.text(0.5, 0.78, str(pred_label),
                transform=ax_text.transAxes, ha='center',
                fontsize=18, fontweight='bold', color='#34495e',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # 2. UNCERTAINTY FLAG (Equally prominent)
    flag_color = '#e74c3c' if uncertainty else '#27ae60'  # Red if uncertain, green if confident
    flag_text = "⚠️ NEEDS REVIEW" if uncertainty else "✓ CONFIDENT"
    
    ax_text.text(0.5, 0.65, flag_text,
                transform=ax_text.transAxes, ha='center',
                fontsize=14, fontweight='bold', color=flag_color,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=flag_color, alpha=0.1, edgecolor=flag_color))
    
    # 3. CONFIDENCE SCORES (Less prominent)
    ax_text.text(0.5, 0.48, "Confidence Scores",
                transform=ax_text.transAxes, ha='center',
                fontsize=10, fontweight='bold', color='#7f8c8d')
    
    ax_text.text(0.5, 0.42, f"Original: {orig_conf:.3f}",
                transform=ax_text.transAxes, ha='center',
                fontsize=9, color='#95a5a6')
    
    ax_text.text(0.5, 0.37, f"Calibrated: {cal_conf:.3f}",
                transform=ax_text.transAxes, ha='center',
                fontsize=9, color='#95a5a6')
    
    # 4. TRUE LABEL (Watermark style at bottom)
    ax_text.text(0.5, 0.08, "Ground Truth Reference",
                transform=ax_text.transAxes, ha='center',
                fontsize=8, style='italic', color='#bdc3c7')
    
    ax_text.text(0.5, 0.05, f"({true_label})",
                transform=ax_text.transAxes, ha='center',
                fontsize=10, style='italic', color='#bdc3c7', alpha=0.9)
    
    # Add border to info panel
    ax_text.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, 
                                   fill=False, edgecolor='#ecf0f1', linewidth=2,
                                   transform=ax_text.transAxes))
    
    plt.tight_layout()
    return fig
    
