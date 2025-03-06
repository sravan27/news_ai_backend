"""
Main application entry point.
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from news_ai_app import __version__
from news_ai_app.api import router
from news_ai_app.config import settings

# Set up logger
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="News AI API",
    description="Advanced news recommendation and analysis API with ML capabilities",
    version=__version__,
    debug=settings.app.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, this should be restricted
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "News AI API",
        "version": __version__,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": __version__}


# Run the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "news_ai_app.main:app",
        host="0.0.0.0",
        port=settings.app.port,
        reload=settings.app.debug
    )