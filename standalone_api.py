#\!/usr/bin/env python3
"""
Standalone API server for News AI application.
Use this when the regular server is experiencing issues.
"""
import os
import sys
import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Make sure the API key environment variables are set
required_vars = ["NEWS_API_KEY", "OPENAI_API_KEY", "HUGGING_FACE_API_KEY"]
missing_vars = [v for v in required_vars if not os.getenv(v)]
if missing_vars:
    logger.error(f"Missing environment variables: {missing_vars}")
    os.environ["NEWS_API_KEY"] = "e9b0f7a3c3004651a35b5f0b042e1828"
    os.environ["OPENAI_API_KEY"] = "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"
    os.environ["HUGGING_FACE_API_KEY"] = "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"
    os.environ["MIND_DATASET_PATH"] = "/Users/sravansridhar/Documents/news_ai/MINDLarge"
    logger.info("Environment variables set automatically")

# Create FastAPI app
app = FastAPI(title="News AI API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

# Import main routes (this needs to be after environment variables are set)
from news_ai_app.api.routes import router
app.include_router(router, prefix="/api")

# Add error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "detail": str(exc)},
    )

if __name__ == "__main__":
    logger.info("Starting standalone News AI API server")
    uvicorn.run("standalone_api:app", host="0.0.0.0", port=8000, reload=True)
