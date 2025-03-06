"""
SQLAlchemy models for the News AI application database.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class User(Base):
    """User model representing application users."""
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    history_count = Column(Integer, default=0)
    interactions = relationship("Interaction", back_populates="user")
    
class NewsArticle(Base):
    """News article model representing news content."""
    __tablename__ = "news_articles"
    
    news_id = Column(String(36), primary_key=True)
    category = Column(String(50))
    subcategory = Column(String(50), nullable=True)
    title = Column(String(256))
    abstract = Column(Text, nullable=True)
    url = Column(String(512), nullable=True)
    entities = Column(JSON, nullable=True)
    publication_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Features and metrics
    embeddings = relationship("Embedding", back_populates="article", uselist=False)
    metrics = relationship("NewsMetrics", back_populates="article", uselist=False)
    interactions = relationship("Interaction", back_populates="article")

class Embedding(Base):
    """Embedding model for storing vector representations of news content."""
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True)
    news_id = Column(String(36), ForeignKey("news_articles.news_id"))
    title_embedding = Column(JSON)  # Store as JSON array
    abstract_embedding = Column(JSON, nullable=True)
    article = relationship("NewsArticle", back_populates="embeddings")

class NewsMetrics(Base):
    """News metrics model for storing calculated metrics for news content."""
    __tablename__ = "news_metrics"
    
    id = Column(Integer, primary_key=True)
    news_id = Column(String(36), ForeignKey("news_articles.news_id"))
    political_influence = Column(Float, nullable=True)
    rhetoric_intensity = Column(Float, nullable=True)  
    information_depth = Column(Float, nullable=True)
    sentiment = Column(Float, nullable=True)
    processing_time = Column(Float, nullable=True)
    article = relationship("NewsArticle", back_populates="metrics")

class Interaction(Base):
    """User interaction model for tracking user-news interactions."""
    __tablename__ = "interactions"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"))
    news_id = Column(String(36), ForeignKey("news_articles.news_id"))
    clicked = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    impression_id = Column(String(36), nullable=True)
    user = relationship("User", back_populates="interactions")
    article = relationship("NewsArticle", back_populates="interactions")

class ModelVersion(Base):
    """Model version management for tracking model versions."""
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100))
    version = Column(String(20))
    uri = Column(String(512))
    metrics = Column(JSON, nullable=True)
    parameters = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=False)
    
class FeatureStore(Base):
    """Feature store for caching and reusing computed features."""
    __tablename__ = "feature_store"
    
    id = Column(Integer, primary_key=True)
    feature_name = Column(String(100))
    entity_id = Column(String(36))  # Can be user_id or news_id
    entity_type = Column(String(20))  # "user" or "news"
    value = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)