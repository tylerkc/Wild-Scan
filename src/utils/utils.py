def parse_s3_uri(s3_uri):
    from urllib.parse import urlparse
    from pathlib import PurePosixPath
    
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
    
