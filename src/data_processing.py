"""
Data Processing Pipeline - October Phase
Cleaning, preprocessing, validation, and chunking of NASA STI data
"""

import pandas as pd
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Tuple


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans, preprocesses, and validates the NASA STI data.
    
    Args:
        df: Raw DataFrame from data acquisition
        
    Returns:
        Preprocessed DataFrame with text_source field ready for chunking
    """
    start_time = time.time()
    print("\n2. Starting Data Cleaning and Preprocessing...")

    # --- Data Cleaning and Validation ---
    df['abstract'] = df['abstract'].fillna("").astype(str)
    df['keywords'] = df['keywords'].apply(lambda x: x if isinstance(x, list) else [])

    initial_count = len(df)
    df = df[df['title'].astype(str).str.strip().str.len() > 0]
    print(f"   - Validation: Filtered {initial_count - len(df)} records with missing titles.")

    # --- Data Feature Engineering (Flattening) ---
    def flatten_authors(affiliations):
        if not isinstance(affiliations, list): 
            return ""
        author_info = []
        for item in affiliations:
            try:
                if 'meta' in item and 'author' in item['meta']:
                    name = item['meta']['author'].get('name', '')
                    if name: 
                        author_info.append(name)
            except:
                continue
        return ", ".join(sorted(list(set(author_info))))

    def list_to_string(item_list):
        if not isinstance(item_list, list): 
            return ""
        return " | ".join(str(item) for item in item_list)

    df['authors_flat'] = df['authorAffiliations'].apply(flatten_authors)
    df['keywords_flat'] = df['keywords'].apply(list_to_string)

    # --- Core Text Generation (The RAG Source) ---
    df['text_source'] = (
        "TITLE: " + df['title'].astype(str) +
        "\nABSTRACT: " + df['abstract'].astype(str) +
        "\nAUTHORS: " + df['authors_flat'].astype(str) +
        "\nKEYWORDS: " + df['keywords_flat'].astype(str)
    )

    df_rag = df[['document_id', 'title', 'abstract', 'text_source']].copy()

    print(f"   ✅ Preprocessing complete. Final record count: {len(df_rag):,}.")
    print(f"   (Preprocessing took {time.time() - start_time:.2f} seconds)")
    return df_rag


def chunk_data(df_rag: pd.DataFrame, chunk_size: int = 512, chunk_overlap: int = 50) -> pd.DataFrame:
    """
    Implements intelligent chunking for titles, abstracts, keywords with optimal size/overlap.
    
    Args:
        df_rag: Preprocessed DataFrame with text_source field
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        DataFrame with chunks ready for embedding
    """
    start_time = time.time()
    print("\n3. Starting Intelligent Chunking...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = []
    total_docs = len(df_rag)

    for index, row in df_rag.iterrows():
        doc_text = row['text_source']
        doc_id = row['document_id']
        doc_title = row['title']

        text_chunks = text_splitter.split_text(doc_text)

        for i, chunk_text in enumerate(text_chunks):
            chunks.append({
                'document_id': doc_id,
                'title': doc_title,
                'chunk_id': f"{doc_id}-{i+1}",
                'chunk_text': chunk_text,
                'chunk_size': len(chunk_text)
            })

    df_chunks = pd.DataFrame(chunks)

    print(f"   ✅ Chunking complete. Total documents processed: {total_docs:,}")
    print(f"   Total chunks created: {len(df_chunks):,}")
    print(f"   Average chunks per document: {len(df_chunks) / total_docs:.2f}")
    print(f"   (Chunking took {time.time() - start_time:.2f} seconds)")
    return df_chunks


def run_processing_pipeline(df_raw: pd.DataFrame, chunk_size: int = 512, chunk_overlap: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the complete data processing pipeline: preprocessing and chunking.
    
    Args:
        df_raw: Raw DataFrame from data acquisition
        chunk_size: Chunk size parameter
        chunk_overlap: Chunk overlap parameter
        
    Returns:
        Tuple of (preprocessed_df, chunks_df)
    """
    df_rag = preprocess_data(df_raw)
    df_chunks = chunk_data(df_rag, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return df_rag, df_chunks

