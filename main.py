from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.models import QuestionRequest, AIResponse, RetrievedDocument
from app.middleware.rate_limit import rate_limit_middleware
from app.core.cache import get_cached_response, cache_response
from app.services.embeddings import get_vectorstore, get_retriever
from app.services.query_gen import get_query_generator, reciprocal_rank_fusion
from app.services.llm import get_answer
from app.core.errors import RAGError, VectorStoreError, QueryGenerationError, LLMError, format_error_response
from app.core.logging import logger
from app.core.langsmith import init_langsmith
import time
from typing import Callable
import traceback
import os
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv('.env.local')

app = FastAPI(title="Chat Genius RAG API")

@app.get("/")
async def root():
    """Root endpoint to verify server is running"""
    return {
        "status": "online",
        "port": os.getenv("PORT", "8000"),
        "environment": os.getenv("RAILWAY_ENVIRONMENT", "local")
    }

# Initialize LangSmith
init_langsmith()

# Configure CORS - single source of truth for CORS handling
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chat-genius-sooty.vercel.app",  # Production
        "http://localhost:3000",                 # Local frontend development
        "http://127.0.0.1:3000",                # Alternative local frontend
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Explicit methods
    allow_headers=[
        "Content-Type",
        "Authorization",
        "Accept",
        "Origin",
        "X-Requested-With",
    ],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Add rate limiting middleware
app.middleware("http")(rate_limit_middleware)

@app.middleware("http")
async def logging_middleware(request: Request, call_next: Callable):
    """Log request and response details."""
    start_time = time.time()
    
    # Log request
    logger.info(
        "Incoming request",
        extra={
            "path": request.url.path,
            "method": request.method,
            "client_ip": request.client.host
        }
    )
    
    try:
        response = await call_next(request)
        
        # Log response
        logger.info(
            "Request completed",
            extra={
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((time.time() - start_time) * 1000, 2)
            }
        )
        return response
        
    except Exception as e:
        # Log error
        logger.error(
            "Request failed",
            extra={
                "path": request.url.path,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "duration_ms": round((time.time() - start_time) * 1000, 2)
            }
        )
        raise

@app.exception_handler(RAGError)
async def rag_error_handler(request: Request, error: RAGError):
    """Handle RAG-specific errors."""
    return JSONResponse(
        status_code=error.status_code,
        content=format_error_response(error)
    )

@app.exception_handler(Exception)
async def general_error_handler(request: Request, error: Exception):
    """Handle unexpected errors."""
    return JSONResponse(
        status_code=500,
        content=format_error_response(error)
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/ask", response_model=AIResponse)
async def ask_question(request: QuestionRequest):
    logger.info(
        "Processing question",
        extra={"question": request.question, "user_id": request.user_id}
    )
    
    try:
        # Check cache first
        cached = get_cached_response(request.question)
        if cached:
            logger.info("Returning cached response")
            # Remove cached field if it exists in the cached data
            if 'cached' in cached:
                del cached['cached']
            return cached
        
        # Get services
        try:
            vectorstore = get_vectorstore()
            retriever = get_retriever(vectorstore)
        except Exception as e:
            raise VectorStoreError(str(e))
        
        # Generate multiple queries
        try:
            query_generator = get_query_generator()
            queries = query_generator.invoke({"original_query": request.question})
            logger.info("Generated queries", extra={"queries": queries})
        except Exception as e:
            raise QueryGenerationError(str(e))
        
        # Get documents for each query
        results = []
        for query in queries:
            try:
                docs = retriever.get_relevant_documents(query)
                results.append(docs)
            except Exception as e:
                raise VectorStoreError(f"Failed to retrieve documents for query '{query}': {str(e)}")
        
        # Rerank documents
        reranked_docs = reciprocal_rank_fusion(results)
        logger.info(
            "Retrieved and reranked documents",
            extra={"num_docs": len(reranked_docs)}
        )
        
        # Generate answer
        try:
            answer = get_answer(request.question, reranked_docs)
        except Exception as e:
            raise LLMError(str(e))
        
        # Format response
        response = AIResponse(
            answer=answer,
            sources=[
                RetrievedDocument(content=doc.page_content, score=score)
                for doc, score in reranked_docs[:3]
            ],
            cached=False
        )
        
        # Cache the response
        cache_response(request.question, response.model_dump())
        
        logger.info("Successfully generated response")
        return response
        
    except Exception as e:
        logger.error(
            "Failed to process question",
            extra={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
        raise

@app.get("/check_env")
def check_env():
    # Be careful not to reveal secrets publicly in production!
    key_exists = bool(os.environ.get("OPENAI_API_KEY"))
    return {"openai_key_loaded": key_exists}

@app.get("/debug/env")
async def debug_env():
    """Debug endpoint to check environment variables (DO NOT USE IN PRODUCTION)"""
    openai_key = os.getenv('OPENAI_API_KEY', 'not_found')
    return {
        "openai_key_exists": bool(openai_key),
        "openai_key_prefix": openai_key[:7] + "..." if openai_key else None,
        "pinecone_key_exists": bool(os.getenv('PINECONE_API_KEY')),
        "langchain_key_exists": bool(os.getenv('LANGCHAIN_API_KEY')),
    }

@app.get("/test-cors")
async def test_cors():
    """Test endpoint to verify CORS configuration"""
    return {"message": "CORS is working!"}

@app.on_event("startup")
async def startup_event():
    """Log diagnostic information on startup"""
    logger.info("Application starting up")
    logger.info(f"PORT env var: {os.getenv('PORT', 'not set')}")
    logger.info(f"RAILWAY_ENVIRONMENT: {os.getenv('RAILWAY_ENVIRONMENT', 'not set')}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info("Configured CORS origins: %s", app.state.cors_origins if hasattr(app.state, 'cors_origins') else 'Not set')

@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown event"""
    logger.info("Application shutting down")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        app,  # Use app directly instead of "main:app"
        host="0.0.0.0",  # Bind to all IPv4 interfaces
        port=port,
        log_level="info",
        proxy_headers=True,  # Trust proxy headers
        forwarded_allow_ips="*",  # Trust forwarded IP headers
        workers=1  # Start with single worker for debugging
    )
