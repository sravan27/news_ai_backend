"""
API routes for the news AI application.
"""
import logging
import time
from typing import Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from openai import OpenAI

from news_ai_app.api.models import (
    BatchMetricsRequest,
    BatchMetricsResponse,
    MetricsRequest,
    MetricsResponse,
    NewsAnalysisRequest,
    NewsAnalysisResponse,
    NewsArticle,
    PodcastGenerationRequest,
    PodcastGenerationResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from news_ai_app.config import settings
from news_ai_app.data import MINDDataset
from news_ai_app.models import HybridNewsRecommender
from news_ai_app.models.advanced_metrics import get_advanced_metrics_calculator
from news_ai_app.utils.news_fetcher import fetch_news_from_url
from news_ai_app.utils.podcast_generator import generate_podcast

# Create router
router = APIRouter()

# Logger
logger = logging.getLogger(__name__)

# Initialize dataset and models
mind_dataset = MINDDataset()
metrics_calculator = get_advanced_metrics_calculator()


# Dependency to get news data
def get_news_data():
    """Get news data from MIND dataset."""
    if mind_dataset.news_df is None:
        mind_dataset.load_news()
    return mind_dataset.news_df


# Dependency to get user behavior data
def get_behaviors_data():
    """Get user behavior data from MIND dataset."""
    if mind_dataset.behaviors_df is None:
        mind_dataset.load_behaviors()
    return mind_dataset.behaviors_df


# Dependency to get entity embeddings
def get_entity_embeddings():
    """Get entity embeddings from MIND dataset."""
    if mind_dataset.entity_embeddings is None:
        mind_dataset.load_entity_embeddings()
    return mind_dataset.entity_embeddings


# Dependency to get relation embeddings
def get_relation_embeddings():
    """Get relation embeddings from MIND dataset."""
    if mind_dataset.relation_embeddings is None:
        mind_dataset.load_relation_embeddings()
    return mind_dataset.relation_embeddings


# Dependency to get recommender model
def get_recommender_model(
    entity_embeddings=Depends(get_entity_embeddings),
    relation_embeddings=Depends(get_relation_embeddings)
):
    """Get recommender model."""
    return HybridNewsRecommender(
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings
    )


@router.get("/news", response_model=List[NewsArticle])
async def get_news(
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    news_df=Depends(get_news_data)
):
    """
    Get news articles, optionally filtered by category and subcategory.
    
    Args:
        category: Filter by news category.
        subcategory: Filter by news subcategory.
        limit: Maximum number of results to return.
        offset: Number of results to skip.
    
    Returns:
        List of news articles.
    """
    # Apply filters
    filtered_df = news_df.copy()
    
    if category:
        filtered_df = filtered_df[filtered_df["category"] == category]
    
    if subcategory:
        filtered_df = filtered_df[filtered_df["subcategory"] == subcategory]
    
    # Apply pagination
    paginated_df = filtered_df.iloc[offset:offset + limit]
    
    # Convert to NewsArticle objects
    news_articles = []
    for _, row in paginated_df.iterrows():
        article = NewsArticle(
            news_id=row["news_id"],
            category=row["category"],
            subcategory=row["subcategory"],
            title=row["title"],
            abstract=row["abstract"],
            url=row["url"],
            title_entities=row["title_entities"],
            abstract_entities=row["abstract_entities"]
        )
        news_articles.append(article)
    
    return news_articles


@router.get("/news/{news_id}", response_model=NewsArticle)
async def get_news_by_id(
    news_id: str,
    news_df=Depends(get_news_data)
):
    """
    Get a news article by its ID.
    
    Args:
        news_id: ID of the news article to retrieve.
    
    Returns:
        News article with the given ID.
    """
    # Find article by ID
    article_df = news_df[news_df["news_id"] == news_id]
    
    if article_df.empty:
        raise HTTPException(status_code=404, detail=f"News article with ID {news_id} not found")
    
    # Get the first (and only) matching article
    row = article_df.iloc[0]
    
    # Convert to NewsArticle object
    article = NewsArticle(
        news_id=row["news_id"],
        category=row["category"],
        subcategory=row["subcategory"],
        title=row["title"],
        abstract=row["abstract"],
        url=row["url"],
        title_entities=row["title_entities"],
        abstract_entities=row["abstract_entities"]
    )
    
    return article


@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    news_df=Depends(get_news_data),
    behaviors_df=Depends(get_behaviors_data),
    recommender=Depends(get_recommender_model)
):
    """
    Get personalized news recommendations.
    
    Args:
        request: Recommendation request containing user ID, history, and preferences.
    
    Returns:
        Recommended news articles with scores.
    """
    start_time = time.time()
    
    # Get user history
    history_news_ids = request.history_news_ids
    
    # If user ID is provided and history is empty, try to get history from behaviors data
    if request.user_id and not history_news_ids:
        user_df = behaviors_df[behaviors_df["user_id"] == request.user_id]
        
        if not user_df.empty:
            # Combine histories from all behavior entries for this user
            for _, row in user_df.iterrows():
                history_news_ids.extend(row["history"])
            
            # Remove duplicates
            history_news_ids = list(dict.fromkeys(history_news_ids))
    
    # If still no history, return error
    if not history_news_ids:
        raise HTTPException(
            status_code=400, 
            detail="No history provided or found for user"
        )
    
    # Get candidate news IDs
    candidate_news_ids = request.candidate_news_ids
    
    # If no candidate IDs provided, use recent news
    if not candidate_news_ids:
        # Sort by most recent (assuming IDs are assigned in order)
        candidate_news_ids = news_df["news_id"].tolist()[-100:]
    
    # Filter out history IDs from candidates to avoid recommending already seen news
    candidate_news_ids = [nid for nid in candidate_news_ids if nid not in history_news_ids]
    
    if not candidate_news_ids:
        raise HTTPException(
            status_code=400, 
            detail="No candidate news items available that user hasn't already seen"
        )
    
    # Get news data for history and candidates
    history_news = news_df[news_df["news_id"].isin(history_news_ids)].to_dict("records")
    candidate_news = news_df[news_df["news_id"].isin(candidate_news_ids)].to_dict("records")
    
    # Get recommendations
    recommendations = recommender.get_recommendations(
        user_history=history_news,
        candidate_news=candidate_news,
        top_k=request.top_k
    )
    
    # Extract scores
    scores = [rec.pop("score") for rec in recommendations]
    
    # Convert to NewsArticle objects
    news_articles = [NewsArticle(**rec) for rec in recommendations]
    
    response = RecommendationResponse(
        user_id=request.user_id,
        recommendations=news_articles,
        scores=scores
    )
    
    logger.info(f"Generated recommendations in {time.time() - start_time:.2f} seconds")
    
    return response


@router.post("/metrics", response_model=MetricsResponse)
async def calculate_metrics(request: MetricsRequest):
    """
    Calculate various metrics for text using advanced ML models.
    
    Args:
        request: Metrics request containing text and metrics to calculate.
    
    Returns:
        Calculated metrics for the text.
    """
    start_time = time.time()
    
    text = request.text
    requested_metrics = request.metrics
    
    # Extract category if available in text
    category = ""
    if "\nCategory: " in text:
        # Try to extract category
        try:
            category_line = [line for line in text.split('\n') if line.startswith("Category: ")][0]
            category = category_line.replace("Category: ", "").strip()
        except:
            pass
    
    # Calculate requested metrics
    metrics_results = {}
    
    if "political_influence" in requested_metrics:
        metrics_results["political_influence"] = metrics_calculator.calculate_political_influence(text, category)
    
    if "rhetoric_intensity" in requested_metrics:
        metrics_results["rhetoric_intensity"] = metrics_calculator.calculate_rhetoric_intensity(text, category)
    
    if "information_depth" in requested_metrics:
        metrics_results["information_depth"] = metrics_calculator.calculate_information_depth(text, category)
    
    if "sentiment" in requested_metrics:
        metrics_results["sentiment"] = metrics_calculator.calculate_sentiment(text, category)
    
    processing_time = time.time() - start_time
    
    return MetricsResponse(
        text=text,
        metrics=metrics_results,
        processing_time=processing_time
    )


@router.post("/metrics/batch", response_model=BatchMetricsResponse)
async def calculate_batch_metrics(request: BatchMetricsRequest):
    """
    Calculate metrics for a batch of texts using advanced ML models.
    
    Args:
        request: Batch metrics request containing texts and metrics to calculate.
    
    Returns:
        Calculated metrics for each text.
    """
    start_time = time.time()
    
    texts = request.texts
    requested_metrics = request.metrics
    
    # Extract categories if available
    categories = []
    for text in texts:
        category = ""
        if "\nCategory: " in text:
            # Try to extract category
            try:
                category_line = [line for line in text.split('\n') if line.startswith("Category: ")][0]
                category = category_line.replace("Category: ", "").strip()
            except:
                pass
        categories.append(category)
    
    # Calculate metrics in batch
    batch_results = metrics_calculator.batch_calculate_metrics(texts, categories)
    
    # Filter requested metrics
    results = []
    for batch_result in batch_results:
        metrics_results = {}
        
        if "political_influence" in requested_metrics and "political_influence" in batch_result:
            metrics_results["political_influence"] = batch_result["political_influence"]
        
        if "rhetoric_intensity" in requested_metrics and "rhetoric_intensity" in batch_result:
            metrics_results["rhetoric_intensity"] = batch_result["rhetoric_intensity"]
        
        if "information_depth" in requested_metrics and "information_depth" in batch_result:
            metrics_results["information_depth"] = batch_result["information_depth"]
        
        if "sentiment" in requested_metrics and "sentiment" in batch_result:
            metrics_results["sentiment"] = batch_result["sentiment"]
        
        results.append(metrics_results)
    
    processing_time = time.time() - start_time
    
    return BatchMetricsResponse(
        results=results,
        processing_time=processing_time
    )


@router.post("/analyze", response_model=NewsAnalysisResponse)
async def analyze_news(request: NewsAnalysisRequest):
    """
    Analyze news content from URL or text.
    
    Args:
        request: News analysis request containing URL or text to analyze.
    
    Returns:
        Analysis results including metrics and entities.
    """
    start_time = time.time()
    
    # Get content from URL or directly from request
    if request.url:
        news_content = await fetch_news_from_url(request.url)
        title = news_content.get("title", "")
        text = news_content.get("text", "")
    else:
        title = request.title or ""
        text = request.text or ""
    
    # Calculate metrics with category if available
    full_text = f"{title} {text}" if title else text
    
    # Extract category from news content if possible
    category = ""
    if news_content and "category" in news_content:
        category = news_content.get("category", "")
        
    metrics_results = metrics_calculator.calculate_all_metrics(full_text, category)
    
    # Extract entities using LangChain and OpenAI
    client = OpenAI(api_key=settings.api_keys.openai_api_key)
    
    entity_prompt = f"""
    Extract named entities from the following text and return them in JSON format.
    For each entity, include:
    - "Label": The entity name
    - "Type": The entity type (P for person, O for organization, G for geographic location, J for product, C for concept, M for media source)
    - "WikidataId": A placeholder ID (use Q1, Q2, etc.)
    - "Confidence": A confidence score between 0 and 1
    - "OccurrenceOffsets": List of position indices where the entity appears
    - "SurfaceForms": List of forms the entity appears as in the text

    Text: {full_text}
    
    Return only the JSON array without any explanation or other text.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": entity_prompt}],
            temperature=0.2
        )
        
        # Extract entities from response
        entities_json = response.choices[0].message.content
        
        # Process entities
        import json
        try:
            entities = json.loads(entities_json)
        except json.JSONDecodeError:
            entities = []
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        entities = []
    
    processing_time = time.time() - start_time
    
    return NewsAnalysisResponse(
        title=title,
        text=text,
        metrics=metrics_results,
        entities=entities,
        processing_time=processing_time
    )


@router.post("/podcast/generate", response_model=PodcastGenerationResponse)
async def generate_podcast_endpoint(
    request: PodcastGenerationRequest,
    news_df=Depends(get_news_data)
):
    """
    Generate a personalized news podcast.
    
    Args:
        request: Podcast generation request containing news IDs and preferences.
    
    Returns:
        Generated podcast URL and metadata.
    """
    start_time = time.time()
    
    # Get news articles
    news_ids = request.news_ids
    news_articles = []
    
    for news_id in news_ids:
        article_df = news_df[news_df["news_id"] == news_id]
        
        if article_df.empty:
            continue
        
        row = article_df.iloc[0]
        article = {
            "news_id": row["news_id"],
            "title": row["title"],
            "abstract": row["abstract"],
            "category": row["category"],
            "subcategory": row["subcategory"]
        }
        
        news_articles.append(article)
    
    if not news_articles:
        raise HTTPException(
            status_code=404, 
            detail="No valid news articles found for the provided IDs"
        )
    
    # Generate podcast
    result = await generate_podcast(
        news_articles=news_articles,
        voice_id=request.voice_id,
        include_intro=request.include_intro,
        include_transitions=request.include_transitions
    )
    
    generation_time = time.time() - start_time
    
    return PodcastGenerationResponse(
        podcast_url=result["url"],
        duration_seconds=result["duration"],
        transcript=result["transcript"],
        news_ids=news_ids,
        generation_time=generation_time
    )