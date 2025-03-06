"""
Pydantic models for API request and response schemas.
"""
from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class Entity(BaseModel):
    """Entity information extracted from news content."""
    label: str
    type: str  
    wikidata_id: str = Field(..., alias="WikidataId")
    confidence: float
    occurrence_offsets: List[int] = Field(..., alias="OccurrenceOffsets")
    surface_forms: List[str] = Field(..., alias="SurfaceForms")


class NewsArticle(BaseModel):
    """News article schema."""
    news_id: str
    category: str
    subcategory: Optional[str] = None
    title: str
    abstract: Optional[str] = None
    url: Optional[str] = None
    title_entities: List[Entity] = []
    abstract_entities: List[Entity] = []
    publication_date: Optional[datetime] = None
    
    class Config:
        populate_by_name = True


class RecommendationRequest(BaseModel):
    """Request body for getting news recommendations."""
    user_id: Optional[str] = None
    history_news_ids: List[str] = Field(default_factory=list)
    candidate_news_ids: Optional[List[str]] = None
    top_k: int = 10
    
    @validator('top_k')
    def validate_top_k(cls, v):
        """Validate that top_k is a positive integer."""
        if v <= 0:
            raise ValueError("top_k must be a positive integer")
        return v


class RecommendationResponse(BaseModel):
    """Response body for news recommendations."""
    user_id: Optional[str] = None
    recommendations: List[NewsArticle]
    scores: List[float]
    timestamps: datetime = Field(default_factory=datetime.now)


class MetricsRequest(BaseModel):
    """Request body for calculating metrics on text."""
    text: str
    metrics: List[str] = ["political_influence", "rhetoric_intensity", "information_depth", "sentiment"]

    
class MetricsResponse(BaseModel):
    """Response body for metrics calculation."""
    text: str
    metrics: Dict[str, Union[float, Dict]]
    processing_time: float


class SentimentResponse(BaseModel):
    """Sentiment analysis result."""
    label: str
    score: float


class BatchMetricsRequest(BaseModel):
    """Request body for batch metrics calculation."""
    texts: List[str]
    metrics: List[str] = ["political_influence", "rhetoric_intensity", "information_depth", "sentiment"]


class BatchMetricsResponse(BaseModel):
    """Response body for batch metrics calculation."""
    results: List[Dict[str, Union[float, Dict]]]
    processing_time: float


class NewsAnalysisRequest(BaseModel):
    """Request body for analyzing external news content."""
    url: Optional[str] = None
    text: Optional[str] = None
    title: Optional[str] = None
    
    @validator('url', 'text')
    def validate_input(cls, v, values):
        """Validate that either url or text is provided."""
        if not values.get('url') and not values.get('text'):
            raise ValueError("Either url or text must be provided")
        return v


class NewsAnalysisResponse(BaseModel):
    """Response body for news content analysis."""
    title: Optional[str] = None
    text: Optional[str] = None
    metrics: Dict[str, Union[float, Dict]]
    entities: List[Entity] = []
    processing_time: float


class PodcastGenerationRequest(BaseModel):
    """Request body for podcast generation."""
    news_ids: List[str]
    user_id: Optional[str] = None
    voice_id: str = "default"
    include_intro: bool = True
    include_transitions: bool = True


class PodcastGenerationResponse(BaseModel):
    """Response body for podcast generation."""
    podcast_url: str
    duration_seconds: float
    transcript: str
    news_ids: List[str]
    generation_time: float