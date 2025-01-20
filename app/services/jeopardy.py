from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from typing import List, Optional, Dict
import os
from app.core.errors import VectorStoreError, LLMError

def get_jeopardy_vectorstore() -> PineconeVectorStore:
    """Get the Jeopardy vectorstore instance."""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        return PineconeVectorStore(
            index_name=os.getenv("NEW_PINECONE_INDEX"),
            embedding=embeddings
        )
    except Exception as e:
        raise VectorStoreError(f"Failed to initialize Jeopardy vectorstore: {str(e)}")

def get_jeopardy_retriever(vectorstore: PineconeVectorStore):
    """Get the retriever for Jeopardy questions."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Get top 5 to allow for filtering
    )

def format_context_from_docs(docs: List[Document]) -> str:
    """Format retrieved Jeopardy questions into context for the LLM."""
    context_parts = []
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        context_parts.append(
            f"Reference {i}:\n"
            f"Category: {metadata.get('category')}\n"
            f"Value: {metadata.get('value')}\n"
            f"Question: {metadata.get('question')}\n"
            f"Answer: {metadata.get('answer')}\n"
        )
    return "\n".join(context_parts)

def get_jeopardy_answer(question: str, context_docs: List[Document]) -> Dict:
    """Generate an answer using the Jeopardy context."""
    try:
        llm = ChatOpenAI(
            model="gpt-4",  # Using GPT-4 for better understanding of Jeopardy format
            temperature=0.7
        )
        
        # Format context
        context = format_context_from_docs(context_docs)
        
        # Create prompt with Jeopardy-specific instructions
        prompt = f"""You are an AI trained to answer questions using Jeopardy! game show questions and answers as reference.
        Use the provided Jeopardy questions and answers as context to answer the user's question.
        If the question is directly related to a Jeopardy clue, format your response in Jeopardy style.
        If it's a general question about topics from Jeopardy, provide a clear, informative answer.
        
        Context from Jeopardy questions:
        {context}
        
        User Question: {question}
        
        Provide your answer in a natural, conversational way while incorporating the Jeopardy knowledge accurately."""
        
        response = llm.invoke(prompt)
        
        # Extract question from content and return filtered metadata
        def extract_metadata(doc: Document) -> dict:
            content_lines = doc.page_content.split('\n')
            question = next((line.split('Question: ')[1] for line in content_lines if 'Question: ' in line), None)
            
            return {
                "air_date": doc.metadata.get("air_date"),
                "category": doc.metadata.get("category"),
                "value": doc.metadata.get("value"),
                "question": question
            }
        
        return {
            "answer": response.content,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": extract_metadata(doc)
                }
                for doc in context_docs[:3]  # Only return top 3 sources
            ]
        }
        
    except Exception as e:
        raise LLMError(f"Failed to generate Jeopardy answer: {str(e)}")

# Placeholder for future chunk support
def get_jeopardy_chunk_vectorstore() -> Optional[PineconeVectorStore]:
    """Get the Jeopardy chunk vectorstore instance (for future use)."""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        return PineconeVectorStore(
            index_name=os.getenv("PINECONE_CHUNK_INDEX"),
            embedding=embeddings
        )
    except Exception:
        return None  # Return None if chunk store isn't available 