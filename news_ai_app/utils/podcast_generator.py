"""
Utilities for generating news podcasts.
"""
import logging
import os
import time
from typing import Dict, List

import httpx
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from openai import OpenAI

from news_ai_app.config import settings

logger = logging.getLogger(__name__)


async def generate_podcast_script(
    news_articles: List[Dict],
    include_intro: bool = True,
    include_transitions: bool = True
) -> str:
    """
    Generate a podcast script from news articles.
    
    Args:
        news_articles: List of news articles to include in the podcast.
        include_intro: Whether to include an introduction.
        include_transitions: Whether to include transitions between articles.
        
    Returns:
        Generated podcast script text.
    """
    try:
        # Prepare the news content for the prompt
        articles_text = ""
        for i, article in enumerate(news_articles):
            articles_text += f"\nARTICLE {i+1}:\n"
            articles_text += f"Title: {article['title']}\n"
            articles_text += f"Category: {article.get('category', 'general')}\n"
            articles_text += f"Content: {article.get('abstract', '')}\n"
            
        # Create the prompt
        prompt_template = """
        You are an experienced news podcast host. Create a engaging and professional news podcast script 
        based on the following articles. The script should sound natural and conversational, not overly formal.
        
        {articles_text}
        
        Guidelines:
        {intro_guideline}
        - For each article, create a compelling summary that captures the key points
        - Do not fabricate details or add speculation not present in the original content
        - Keep each article section to approximately 1-2 minutes when read aloud
        - Use a conversational tone with clear transitions between topics
        {transition_guideline}
        - End with a brief conclusion
        
        Generate the complete podcast script:
        """
        
        # Add conditional guidelines
        intro_guideline = "- Begin with a short introduction welcoming listeners to the news podcast" if include_intro else ""
        transition_guideline = "- Create natural transitions between articles that connect the topics when possible" if include_transitions else ""
        
        # Replace placeholders in the prompt template
        prompt = prompt_template.format(
            articles_text=articles_text,
            intro_guideline=intro_guideline,
            transition_guideline=transition_guideline
        )
        
        # Use OpenAI to generate the script
        client = OpenAI(api_key=settings.api_keys.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error generating podcast script: {e}")
        return f"Could not generate podcast script: {str(e)}"


async def convert_text_to_speech(text: str, voice_id: str = "default") -> Dict:
    """
    Convert text to speech using text-to-speech API.
    
    This would typically use a proper TTS service like ElevenLabs or Amazon Polly.
    For now, we'll mock this functionality.
    
    Args:
        text: The text to convert to speech.
        voice_id: The voice to use for the conversion.
        
    Returns:
        Dictionary with URL to audio file and duration.
    """
    try:
        # In a real implementation, this would call a TTS API
        # For now, we'll simulate this process
        
        # Simulate processing time
        await httpx.AsyncClient().sleep(1.0)
        
        # Calculate an estimated duration based on reading speed
        # Average reading speed is about 150 words per minute
        word_count = len(text.split())
        estimated_duration = (word_count / 150) * 60  # in seconds
        
        # Generate a mock URL
        timestamp = int(time.time())
        mock_audio_url = f"https://example.com/podcast/generated-{timestamp}.mp3"
        
        return {
            "url": mock_audio_url,
            "duration": estimated_duration
        }
        
    except Exception as e:
        logger.error(f"Error converting text to speech: {e}")
        return {
            "url": "",
            "duration": 0,
            "error": str(e)
        }


async def generate_podcast(
    news_articles: List[Dict],
    voice_id: str = "default",
    include_intro: bool = True,
    include_transitions: bool = True
) -> Dict:
    """
    Generate a complete podcast from news articles.
    
    Args:
        news_articles: List of news articles to include in the podcast.
        voice_id: The voice ID to use for text-to-speech conversion.
        include_intro: Whether to include an introduction.
        include_transitions: Whether to include transitions between articles.
        
    Returns:
        Dictionary with podcast URL, duration, and transcript.
    """
    try:
        # Generate the podcast script
        transcript = await generate_podcast_script(
            news_articles=news_articles,
            include_intro=include_intro,
            include_transitions=include_transitions
        )
        
        # Convert the script to speech
        speech_result = await convert_text_to_speech(
            text=transcript,
            voice_id=voice_id
        )
        
        return {
            "url": speech_result["url"],
            "duration": speech_result["duration"],
            "transcript": transcript
        }
        
    except Exception as e:
        logger.error(f"Error generating podcast: {e}")
        return {
            "url": "",
            "duration": 0,
            "transcript": "",
            "error": str(e)
        }