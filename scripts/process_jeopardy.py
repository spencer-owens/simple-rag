import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from typing import List, Dict, Optional
import itertools
import time
from datetime import datetime

# Load environment variables
load_dotenv(".env.local")

# Set environment variables
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_ENVIRONMENT"] = os.getenv("PINECONE_ENVIRONMENT")

# Constants
JEOPARDY_SMALL_INDEX = os.getenv("NEW_PINECONE_INDEX")
JEOPARDY_CHUNK_INDEX = os.getenv("PINECONE_CHUNK_INDEX")
BATCH_SIZE = 100
CHUNK_SIZE = 5  # Number of questions per chunk

def clean_value(value: any) -> str:
    """Clean and format the value field."""
    if pd.isna(value) or value is None:
        return "Unknown"
    if isinstance(value, (int, float)):
        return f"${int(value)}"
    # Remove any non-numeric characters and convert to integer
    try:
        numeric = ''.join(filter(str.isdigit, str(value)))
        if numeric:
            return f"${numeric}"
    except:
        pass
    return str(value).strip()

def clean_text(text: any) -> str:
    """Clean and format text fields."""
    if pd.isna(text) or text is None:
        return "Unknown"
    return str(text).strip()

def validate_row(row: pd.Series) -> bool:
    """Validate if a row has all required fields with valid data."""
    required_fields = ['Category', 'Question', 'Answer']
    
    for field in required_fields:
        if field not in row:
            return False
        if pd.isna(row[field]) or row[field] is None:
            return False
        if not str(row[field]).strip():
            return False
    
    return True

def format_jeopardy_text(row: pd.Series) -> Optional[str]:
    """Format a Jeopardy question into a standardized text format."""
    if not validate_row(row):
        return None
        
    return (f"Category: {clean_text(row['Category'])}\n"
            f"Value: {clean_value(row['Value'])}\n"
            f"Question: {clean_text(row['Question'])}\n"
            f"Answer: {clean_text(row['Answer'])}\n"
            f"Round: {clean_text(row['Round'])}")

def create_metadata(row: pd.Series) -> Dict:
    """Create metadata for a Jeopardy question."""
    return {
        "show_number": clean_text(row["Show Number"]),
        "air_date": clean_text(row["Air Date"]),
        "category": clean_text(row["Category"]),
        "value": clean_value(row["Value"]),
        "round": clean_text(row["Round"]),
        "answer": clean_text(row["Answer"])
    }

def create_chunk_text(questions: List[Dict]) -> str:
    """Create a combined text from multiple questions."""
    texts = []
    for i, q in enumerate(questions, 1):
        formatted = format_jeopardy_text(pd.Series(q))
        if formatted:
            texts.append(f"Question {i}:\n{formatted}\n")
    return "\n".join(texts)

def create_chunk_metadata(questions: List[Dict]) -> Dict:
    """Create metadata for a chunk of questions."""
    return {
        "show_numbers": [clean_text(q["Show Number"]) for q in questions],
        "categories": list(set(clean_text(q["Category"]) for q in questions)),
        "question_count": len(questions)
    }

def process_batch(batch_df: pd.DataFrame, embeddings, small_index: bool = True) -> List[Document]:
    """Process a batch of questions for either small or chunk index."""
    documents = []
    skipped = 0
    
    if small_index:
        # Process individual questions for small index
        for _, row in batch_df.iterrows():
            text = format_jeopardy_text(row)
            if text:  # Only process valid questions
                metadata = create_metadata(row)
                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)
            else:
                skipped += 1
    else:
        # Process chunks for chunk index
        valid_rows = [row for _, row in batch_df.iterrows() if validate_row(row)]
        for i in range(0, len(valid_rows), CHUNK_SIZE):
            chunk_rows = valid_rows[i:i + CHUNK_SIZE]
            if chunk_rows:  # Only process non-empty chunks
                text = create_chunk_text([row.to_dict() for row in chunk_rows])
                if text:  # Verify we have valid text after processing
                    metadata = create_chunk_metadata([row.to_dict() for row in chunk_rows])
                    doc = Document(page_content=text, metadata=metadata)
                    documents.append(doc)
                else:
                    skipped += 1
    
    if skipped > 0:
        print(f"Skipped {skipped} invalid items in batch")
    
    return documents

def main():
    start_time = time.time()
    total_questions = 0
    total_chunks = 0
    error_count = 0
    max_retries = 3
    skipped_total = 0

    print(f"Starting Jeopardy data processing at {datetime.now()}")
    print(f"Small Index: {JEOPARDY_SMALL_INDEX}")
    print(f"Chunk Index: {JEOPARDY_CHUNK_INDEX}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Chunk Size: {CHUNK_SIZE}")
    print("=" * 80)

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Process the CSV in batches
    for batch_num, batch_df in enumerate(pd.read_csv("JEOPARDY_CSV.csv", 
                                                    chunksize=BATCH_SIZE,
                                                    skipinitialspace=True,
                                                    keep_default_na=False)):  # Prevent automatic NA conversion
        batch_start = time.time()
        print(f"\nProcessing batch {batch_num + 1}")
        
        try:
            # Process for small index (individual questions)
            small_documents = process_batch(batch_df, embeddings, small_index=True)
            if small_documents:  # Only upload if we have valid documents
                PineconeVectorStore.from_documents(
                    documents=small_documents,
                    embedding=embeddings,
                    index_name=JEOPARDY_SMALL_INDEX
                )
            
            # Process for chunk index (grouped questions)
            chunk_documents = process_batch(batch_df, embeddings, small_index=False)
            if chunk_documents:  # Only upload if we have valid documents
                PineconeVectorStore.from_documents(
                    documents=chunk_documents,
                    embedding=embeddings,
                    index_name=JEOPARDY_CHUNK_INDEX
                )
            
            batch_time = time.time() - batch_start
            total_questions += len(small_documents)
            total_chunks += len(chunk_documents)
            skipped_batch = len(batch_df) - len(small_documents)
            skipped_total += skipped_batch
            
            print(f"Completed batch {batch_num + 1} in {batch_time:.2f}s:")
            print(f"  - Questions: {len(small_documents)} (Total: {total_questions})")
            print(f"  - Chunks: {len(chunk_documents)} (Total: {total_chunks})")
            print(f"  - Skipped: {skipped_batch} (Total: {skipped_total})")
            
        except Exception as e:
            error_count += 1
            print(f"Error processing batch {batch_num + 1}: {str(e)}")
            if error_count >= max_retries:
                print("Max errors reached. Stopping processing.")
                break
            print("Continuing with next batch...")
            continue

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Processing completed at {datetime.now()}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total questions processed: {total_questions}")
    print(f"Total chunks processed: {total_chunks}")
    print(f"Total questions skipped: {skipped_total}")
    print(f"Total errors: {error_count}")

if __name__ == "__main__":
    main() 