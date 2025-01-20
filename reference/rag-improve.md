### 1. Prepare the Jeopardy Data

- Ensure you have your Jeopardy data in either CSV or JSON format (whichever is most convenient).

- If the data is large, consider segmenting it into manageable documents (e.g., each question/answer pair as a “document”), so that each record can be embedded separately.

---

### 2. Create a New Pinecone Index

- Log in to your Pinecone dashboard and create a new index (e.g., “jeopardy-rag-index”). Make sure it has enough dimensions to handle your chosen embeddings (e.g., if using “text-embedding-ada-002”, then 1536 dimensions).

- Note your new index name. Update your .env file accordingly (e.g., PINECONE_INDEX_JEOPARDY) and do NOT store the Pinecone API key in any non-.env file.

---

### 3. Create or Update an Embedding Script

Much like you have a scripts/add_embeddings.py, you can create a new script specifically for the Jeopardy data ingestion. For example, “scripts/add_jeopardy_embeddings.py”.

Below is an example reference for your script (notice no line numbers are included):

add_jeopardy_embeddings.py

Apply

import os

from dotenv import load_dotenv

from pathlib import Path

from langchain_openai import OpenAIEmbeddings

from pinecone import Pinecone

def load_jeopardy_data():

    # TODO: Implement CSV or JSON reading logic

    # Return data as a dict: {doc_id: text, doc_id2: text, ...}

    return {

        "doc1": "Sample Jeopardy clue and answer data..."

    }

def main():

    # Load environment

    api_dir = Path(__file__).resolve().parent.parent

    env_path = api_dir / ".env.local"

    load_dotenv(env_path)

    # Initialize Pinecone

    api_key = os.getenv("PINECONE_API_KEY")

    pc = Pinecone(api_key=api_key)

    # Create/Open desired index

    index_name = os.getenv("PINECONE_INDEX_JEOPARDY", "jeopardy-rag-index")

    index = pc.Index(index_name)

    # Prepare data

    all_documents = load_jeopardy_data()

    # Create embeddings

    embeddings = OpenAIEmbeddings()

    # Upsert to Pinecone

    for doc_id, text in all_documents.items():

        vector = embeddings.embed_query(text)

        index.upsert([(doc_id, vector, {"text": text})])

    print("Finished uploading Jeopardy embeddings to Pinecone.")

if __name__ == "__main__":

    main()

Notes:

- Replace the “load_jeopardy_data()” logic to parse your actual CSV or JSON.

- It’s a good practice to keep this file under ~250 lines by splitting out any large functions.

---

### 4. Update Your RAG Configuration

- In “app/core/config.py”, add a new environment variable for your Jeopardy Pinecone index (e.g., pinecone_index_jeopardy).

- In “app/services/embeddings.py” (or a new module if you prefer), create a function similar to get_vectorstore() that loads your Jeopardy index. For instance:
    
    embeddings.py
    
    Apply
    
       def get_jeopardy_vectorstore():
    
           embeddings = OpenAIEmbeddings()
    
           pc = PineconeClient(api_key=settings.pinecone_api_key)
    
           index = pc.Index(settings.pinecone_index_jeopardy)
    
           return Pinecone.from_existing_index(
    
               index_name=settings.pinecone_index_jeopardy,
    
               embedding=embeddings,
    
               namespace=""
    
           )
    

- If you want to switch between your old index and the new Jeopardy index at runtime, you can create a query parameter or environment-based logic to decide which vectorstore to call.

---

### 5. Update or Expand LLM Logic for Jeopardy Context

• Currently, your RAG logic lives in “main.py” under the /ask endpoint, where:

- A user question is passed to get_vectorstore → get_retriever → get_relevant_documents.

- The reciprocal_rank_fusion merges docs.

- get_answer formats the prompt.

• If your new Jeopardy-based corpus demands a different prompt or a different approach (e.g., you might want a smaller temperature or a different style), you can either:

- Update the existing get_answer() logic, or

- Create a new function with a specialized prompt.

- Make sure to keep the “system” instructions and user prompt relevant to Jeopardy questions/answers.

---

### 6. Update LangSmith Project

If you want a new project name in LangSmith, you can do so in “app/core/langsmith.py” or wherever you set your environment variables:

• In your .env (or .env.local), add:

LANGCHAIN_PROJECT=jeopardy-observations

• Then in code:

- The init_langsmith() function can optionally log that environment variable or create the new project if you want.

- LangSmith usage should largely remain the same; it will simply group new traces into the new project name.

---

### 7. Testing and Verification

- Run your new “add_jeopardy_embeddings.py” script to populate the Pinecone index with Jeopardy data.

- Start your FastAPI server.

- Query the new endpoint. Ensure you’re retrieving Jeopardy questions and answers.

- Review logs and LangSmith traces to confirm the new project is recognized.
