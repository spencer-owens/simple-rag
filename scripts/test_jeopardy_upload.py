import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

# Load environment variables
load_dotenv(".env.local")

# Set environment variables
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_ENVIRONMENT"] = os.getenv("PINECONE_ENVIRONMENT")

# Constants
JEOPARDY_SMALL_INDEX = os.getenv("NEW_PINECONE_INDEX")
JEOPARDY_CHUNK_INDEX = os.getenv("PINECONE_CHUNK_INDEX")
TEST_SAMPLE_SIZE = 5

def format_jeopardy_text(row):
    """Format a Jeopardy question into a standardized text format."""
    return (f"Category: {row['Category'].strip()}\n"
            f"Value: {row['Value'].strip()}\n"
            f"Question: {row['Question'].strip()}\n"
            f"Answer: {row['Answer'].strip()}\n"
            f"Round: {row['Round'].strip()}")

def create_metadata(row):
    """Create metadata for a Jeopardy question."""
    return {
        "show_number": str(row["Show Number"]).strip(),
        "air_date": row["Air Date"].strip(),
        "category": row["Category"].strip(),
        "value": row["Value"].strip(),
        "round": row["Round"].strip(),
        "answer": row["Answer"].strip()
    }

def main():
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Read sample of Jeopardy data
    df = pd.read_csv("JEOPARDY_CSV.csv", 
                     nrows=TEST_SAMPLE_SIZE,
                     skipinitialspace=True)  # Skip spaces after commas
    print(f"Loaded {len(df)} sample questions")
    print("Columns:", df.columns.tolist())

    # Create documents for individual questions
    documents = []
    for _, row in df.iterrows():
        text = format_jeopardy_text(row)
        metadata = create_metadata(row)
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)

    # Upload to jeopardy-small index
    print(f"Uploading {len(documents)} documents to {JEOPARDY_SMALL_INDEX}")
    vectorstore_small = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=JEOPARDY_SMALL_INDEX
    )

    # Test retrieval
    test_query = "Who was Jim Thorpe?"
    results = vectorstore_small.similarity_search(
        test_query,
        k=2
    )
    
    print("\nTest Results:")
    for doc in results:
        print("\nRetrieved Document:")
        print(doc.page_content)
        print("\nMetadata:", doc.metadata)

if __name__ == "__main__":
    main() 