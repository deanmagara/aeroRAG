"""
Data Acquisition Module - September Phase
Downloads and loads NASA STI Repository NDJSON datasets
"""

import json
import pandas as pd
import gzip
import requests
import io
import time
from typing import Optional, Dict, Any


def load_data_from_url(url: str) -> Optional[pd.DataFrame]:
    """
    Downloads, decompresses, and loads the NASA STI metadata from URL.
    
    Args:
        url: URL to the compressed JSON file
        
    Returns:
        DataFrame with NASA STI records or None if error
    """
    start_time = time.time()
    print(f"\n1. Starting Data Acquisition and Loading from URL...")
    try:
        # 1. Download the file content
        print("   📥 Downloading compressed file...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # 2. Decompress
        print("   📦 Decompressing gzip file...")
        compressed_file = io.BytesIO(response.content)

        # Use gzip to open the compressed content as text
        with gzip.open(compressed_file, 'rt', encoding='utf-8') as f:
            # 3. Load the single JSON object
            print("   📖 Parsing JSON data...")
            data = json.load(f)

        print(f"   ✅ Data loaded. Type: {type(data)}. Size: {len(data):,} items.")

        # 4. Convert to DataFrame
        records = []
        if isinstance(data, dict):
            for doc_id, doc_data in data.items():
                if isinstance(doc_data, dict):
                    record = doc_data.copy()
                    record['document_id'] = doc_id
                    records.append(record)

            df = pd.DataFrame(records)
            print(f"   ✅ Converted to DataFrame with {len(df):,} rows.")
            return df
        else:
            print("   ❌ Error: Data root structure is not a dictionary.")
            return None

    except requests.RequestException as e:
        print(f"   ❌ ERROR: Network/Download error: {e}")
        return None
    except (gzip.BadGzipFile, json.JSONDecodeError) as e:
        print(f"   ❌ ERROR: Decompression or JSON parsing error: {e}")
        return None
    except Exception as e:
        print(f"   ❌ An unexpected error occurred: {e}")
        return None
    finally:
        print(f"   (Data Acquisition took {time.time() - start_time:.2f} seconds)")


def load_data_from_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    Loads NASA STI metadata from a local JSON file.
    
    Args:
        file_path: Path to local JSON file
        
    Returns:
        DataFrame with NASA STI records or None if error
    """
    start_time = time.time()
    print(f"1. Loading data from {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"   ✅ Data loaded. Type: {type(data)}. Size: {len(data):,} items.")
        
        records = []
        if isinstance(data, dict):
            for doc_id, doc_data in data.items():
                if isinstance(doc_data, dict):
                    record = doc_data.copy()
                    record['document_id'] = doc_id 
                    records.append(record)
            
            df = pd.DataFrame(records)
            print(f"   ✅ Converted to DataFrame with {len(df):,} rows.")
            return df
        
        elif isinstance(data, list):
            df = pd.DataFrame(data)
            print(f"   ✅ Converted to DataFrame with {len(df):,} rows.")
            return df

    except FileNotFoundError:
        print(f"   ❌ ERROR: File not found at '{file_path}'. Please check the path.")
        return None
    except json.JSONDecodeError as e:
        print(f"   ❌ ERROR: JSON Decode Error: {e}. Check file integrity.")
        return None
    except Exception as e:
        print(f"   ❌ An unexpected error occurred during loading: {e}")
        return None
    finally:
        print(f"   (Loading took {time.time() - start_time:.2f} seconds)")

