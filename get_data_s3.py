import os
import json
import logging

import boto3
import pandas as pd
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Retrieve the S3 bucket name
BUCKET_NAME = os.getenv("BUCKET_NAME")
PREFIX = os.getenv("PREFIX")

# Initialize S3 client
s3 = boto3.client("s3")

def list_s3_files(bucket_name, prefix=""):
    """List all JSON files within a given S3 prefix."""
    files = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if "Contents" in page:
            files.extend([obj["Key"] for obj in page["Contents"] if obj["Key"].endswith(".json")])
    return files

def fetch_json_from_s3(bucket_name, file_key):
    """Fetch and parse JSON data from S3."""
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return data
    except Exception as e:
        logger.error(f"Error fetching {file_key}: {e}")
        return None

# Fetch all JSON files under "candles/15min/year=2025/"
logger.info("Fetching JSON files")
json_files = list_s3_files(BUCKET_NAME, PREFIX)

# Load data into a Pandas DataFrame
data_list = []
for file in json_files:
    json_data = fetch_json_from_s3(BUCKET_NAME, file)
    if json_data:
        # Select relevant fields
        filtered_data = {
            "timestamp": json_data["timestamp"],
            "open": json_data["open"],
            "high": json_data["high"],
            "low": json_data["low"],
            "close": json_data["close"]
        }
        data_list.append(filtered_data)

# Convert to DataFrame
df = pd.DataFrame(data_list)

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort by timestamp
df = df.sort_values(by="timestamp")

# Display DataFrame
print(len(df))

df.to_csv("sol_price.csv")
