import json
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import time

# --- Configuration ---
FILE_PATH = "ntrs-public-metadata.json"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# --- Helper Function: Load Data ---
def load_data(file_path):
    """Loads a single, standard JSON file."""
    start_time = time.time()
    print(f"1. Loading data from {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"   ✅ Data loaded. Type: {type(data)}. Size: {len(data):,} items.")
        
        # Convert the dictionary (where keys are IDs) into a list of records for a DataFrame
        records = []
        if isinstance(data, dict):
            for doc_id, doc_data in data.items():
                if isinstance(doc_data, dict):
                    record = doc_data.copy()
                    # Preserve the ID as a primary column
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


# --- Step 1: Data Cleaning and Preprocessing ---
def preprocess_data(df):
    """Cleans, preprocesses, and validates the data."""
    start_time = time.time()
    print("\n2. Starting Data Cleaning and Preprocessing...")

    # --- Data Cleaning and Validation ---
    
    # Fill missing 'abstract' and 'keywords'
    df['abstract'] = df['abstract'].fillna("").astype(str)
    # Ensure keywords is a list for processing, fill non-lists with empty list
    df['keywords'] = df['keywords'].apply(lambda x: x if isinstance(x, list) else [])
    
    # Basic validation: filter out records with no title
    initial_count = len(df)
    df = df[df['title'].astype(str).str.strip().str.len() > 0]
    print(f"   - Validation: Filtered {initial_count - len(df)} records with missing titles.")
    
    # --- Data Feature Engineering (Flattening) ---
    
    # Function to flatten nested author data
    def flatten_authors(affiliations):
        if not isinstance(affiliations, list): return ""
        author_info = []
        for item in affiliations:
            try:
                if 'meta' in item and 'author' in item['meta']:
                    name = item['meta']['author'].get('name', '')
                    if name: author_info.append(name)
            except:
                continue
        return ", ".join(sorted(list(set(author_info))))

    # Function to flatten list of strings/dicts (like keywords) to a string
    def list_to_string(item_list):
        if not isinstance(item_list, list): return ""
        return " | ".join(str(item) for item in item_list)

    # Apply the flattening functions
    df['authors_flat'] = df['authorAffiliations'].apply(flatten_authors)
    df['keywords_flat'] = df['keywords'].apply(list_to_string)
    
    # --- Core Text Generation (The RAG Source) ---
    df['text_source'] = (
        "TITLE: " + df['title'].astype(str) + 
        "\nABSTRACT: " + df['abstract'].astype(str) + 
        "\nAUTHORS: " + df['authors_flat'].astype(str) +
        "\nKEYWORDS: " + df['keywords_flat'].astype(str)
    )
    
    # Select only the essential columns for the next stage
    df_rag = df[['document_id', 'title', 'abstract', 'text_source']].copy()
    
    print(f"   ✅ Preprocessing complete. Final record count: {len(df_rag):,}.")
    print(f"   Sample Combined Text (First 300 chars):")
    print(df_rag['text_source'].iloc[0][:300].replace('\n', ' '))
    print(f"   (Preprocessing took {time.time() - start_time:.2f} seconds)")
    return df_rag


# --- Step 2: Chunking Strategy Implementation ---
def chunk_data(df_rag):
    """Implements intelligent chunking using RecursiveCharacterTextSplitter."""
    start_time = time.time()
    print("\n3. Starting Intelligent Chunking...")

    # 1. Initialize the Recursive Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Prioritize keeping semantic units together
        separators=["\n\n", "\n", ".", " ", ""] 
    )

    # 2. Process each document and create chunks
    chunks = []
    total_docs = len(df_rag)
    
    for index, row in df_rag.iterrows():
        doc_text = row['text_source']
        doc_id = row['document_id']
        doc_title = row['title']
        
        # Split the document text into chunks
        text_chunks = text_splitter.split_text(doc_text)
        
        # Create the chunk records
        for i, chunk_text in enumerate(text_chunks):
            chunks.append({
                'document_id': doc_id,
                'title': doc_title,
                'chunk_id': f"{doc_id}-{i+1}",
                'chunk_text': chunk_text,
                'chunk_size': len(chunk_text)
            })

    # Convert the list of chunks into a new DataFrame
    df_chunks = pd.DataFrame(chunks)

    print(f"   ✅ Chunking complete. Total documents processed: {total_docs:,}")
    print(f"   Total chunks created: {len(df_chunks):,}")
    print(f"   (Chunking took {time.time() - start_time:.2f} seconds)")
    return df_chunks


# --- Step 3: Pipeline Documentation and Execution ---
def run_pipeline():
    """Executes the full pipeline and provides documentation."""
    
    pipeline_doc = f"""
============================================================
PIPELINE DOCUMENTATION: NASA NTRS METADATA RAG PREPARATION
============================================================

1. DATA SOURCE & ACQUISITION:
-----------------------------
- Source: Local file '{FILE_PATH}'.
- Format: Uncompressed JSON (single dictionary/object).
- Acquisition Method: Python 'json.load'.

2. DATA CLEANING, PREPROCESSING & VALIDATION:
---------------------------------------------
- **Cleaning/Flattening**: Nested fields ('authorAffiliations', 'keywords') converted to simple, pipe/comma-separated strings.
- **Validation**: Records with missing titles are dropped.
- **Core Text Generation**: A master field (`text_source`) is created for RAG by concatenating TITLE, ABSTRACT, AUTHORS, and KEYWORDS.

3. CHUNKING STRATEGY:
---------------------
- **Tool**: RecursiveCharacterTextSplitter (LangChain).
- **Goal**: Maintain semantic boundaries by prioritizing splits at paragraphs and sentences.
- **Parameters**:
    - Chunk Size: {CHUNK_SIZE} characters (Optimal for LLM context).
    - Chunk Overlap: {CHUNK_OVERLAP} characters (Ensures context continuity).

4. OUTPUT:
----------
- The final output is a DataFrame (`df_chunks`) ready for vector embedding, where each row is a semantic chunk.
============================================================
"""
    print(pipeline_doc)
    
    # Execution
    df_raw = load_data(FILE_PATH)
    if df_raw is None:
        return

    df_rag = preprocess_data(df_raw)
    
    df_chunks = chunk_data(df_rag)
    
    # Optional: Save the chunks to a file for the next step (embedding/vector storage)
    # print("\n4. Saving final chunks to 'ntrs_chunks.csv'...")
    # df_chunks.to_csv("ntrs_chunks.csv", index=False)
    # print("   ✅ Save complete.")

    print(f"\n--- Final Output DataFrame Head (First Chunk) ---")
    print(df_chunks.head(1).T)
    print(f"\nPipeline Execution Complete!")


if __name__ == "__main__":
    run_pipeline()