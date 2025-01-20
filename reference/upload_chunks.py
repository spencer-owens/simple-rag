from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
PINECONE_CHUNK_INDEX = os.getenv("PINECONE_CHUNK_INDEX")

embeddings = OpenAIEmbeddings()

loader = DirectoryLoader('./docs', glob="**/*.pdf", loader_cls=PyPDFLoader)
raw_docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
documents = text_splitter.split_documents(raw_docs)
print(f"Adding {len(documents)} chunks to Pinecone")

PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=PINECONE_CHUNK_INDEX)

print("All chunks processed")
