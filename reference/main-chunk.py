from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
PINECONE_SUMMARY_INDEX = os.getenv("PINECONE_SUMMARY_INDEX")
PINECONE_CHUNK_INDEX = os.getenv("PINECONE_CHUNK_INDEX")

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
embeddings = OpenAIEmbeddings()

prompt = "Tell me some of the lessons learned in warfare in the last few hundred years"

summary_vectorstore = PineconeVectorStore(index_name=PINECONE_SUMMARY_INDEX, embedding=embeddings)
chunk_vectorstore = PineconeVectorStore(index_name=PINECONE_CHUNK_INDEX, embedding=embeddings)

summary_retriever = summary_vectorstore.as_retriever()
summary_result = summary_retriever.invoke(prompt)[0]
summary_source = summary_result.metadata['source']

print(f"Top Summary Source: {summary_source}")
print(f"Summary Content: {summary_result.page_content}\n\n")

chunk_retriever = chunk_vectorstore.as_retriever(filter={"source": summary_source})
chunk_results = chunk_retriever.invoke(prompt, top_k=4)

chunk_content = "\n\n".join([doc.page_content for doc in chunk_results])

print(f"Chunk Content from {summary_source}:\n{chunk_content}\n\n")

template = PromptTemplate(template="{query} Context: {context}", input_variables=["query", "context"])
prompt_with_context = template.invoke({"query": prompt, "context": chunk_content})

response = llm.invoke(prompt_with_context)

print(response.content)
