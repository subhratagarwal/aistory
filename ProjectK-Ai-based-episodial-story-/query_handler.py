import openai
import json
import os
from config import (USER_CONFIG, get_model_provider, get_openai_api_key, 
                   get_model_names, get_ollama_base_url, MAX_MEMORY_EPISODES)
import logging
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory class to create appropriate model handler based on configuration."""
    _instance = None
    
    @classmethod
    def get_handler(cls):
        """Get model handler, recreating it each time to ensure latest config is used."""
        provider = get_model_provider()
        
        if provider == "openai":
            return OpenAIHandler()
        elif provider == "ollama":
            return OllamaHandler()
        else:
            logger.error(f"Unknown model provider: {provider}")
            raise ValueError(f"Unknown model provider: {provider}")

class ModelHandler:
    """Abstract base class for model handlers."""
    def generate_response(self, prompt, model_type="small", temperature=0.7, max_tokens=None):
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_model_name(self, model_type):
        return get_model_names(model_type)

class OpenAIHandler(ModelHandler):
    """Handler for OpenAI API models."""
    def __init__(self):
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError("OpenAI API key is not set")
        self.client = openai.Client(api_key=api_key)
        self.cache = {}  # Simple in-memory cache
    
    def generate_response(self, prompt, model_type="small", temperature=0.7, max_tokens=None):
        model = self.get_model_name(model_type)
        
        # Create cache key
        cache_key = f"{model}:{prompt[:50]}:{temperature}:{max_tokens}"
        
        # Check cache
        if cache_key in self.cache:
            logger.info(f"Cache hit for query with model {model}")
            return self.cache[cache_key]
        
        try:
            # Set default max tokens based on model
            if not max_tokens:
                max_tokens = 1500 if model_type == "small" else 3000
            
            # Create system prompt
            system_prompt = self._get_system_prompt()
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = response.choices[0].message.content
            
            # Cache the result
            self.cache[cache_key] = result
            
            return result
        except Exception as e:
            logger.error(f"Error generating response with {model}: {str(e)}")
            return f"Error: {str(e)}"
    
    def _get_system_prompt(self):
        return """You are an expert radio dramatist and storyteller creating emotionally engaging narrative content for audio.

Your writing specialties include:
1. Creating distinctive character voices that are identifiable through dialogue alone
2. Crafting sound-rich scenes that build a theater of the mind
3. Developing emotional moments that resonate deeply with listeners
4. Building tension and release with careful pacing and hooks
5. Using narration and dialogue effectively for audio storytelling

Your writing is detailed, emotionally resonant, and maintains consistent character voices throughout the narrative."""

class OllamaHandler(ModelHandler):
    """Handler for Ollama local models."""
    def __init__(self):
        self.cache = {}  # Simple in-memory cache
        self.small_model = None
        self.big_model = None
    
    def _initialize_models(self):
        """Lazy initialization of models."""
        try:
            base_url = get_ollama_base_url()
            small_model_name = self.get_model_name("small")
            big_model_name = self.get_model_name("big")
            
            # Always reinitialize to ensure we're using current config
            self.small_model = OllamaLLM(
                model=small_model_name,
                base_url=base_url,
                system=self._get_system_prompt()
            )
            
            self.big_model = OllamaLLM(
                model=big_model_name,
                base_url=base_url,
                system=self._get_system_prompt()
            )
            
            logger.info(f"Initialized Ollama models: small={small_model_name}, big={big_model_name}")
        except Exception as e:
            logger.error(f"Error initializing Ollama models: {str(e)}")
            raise e
    
    def generate_response(self, prompt, model_type="small", temperature=0.7, max_tokens=None):
        self._initialize_models()
        model_name = self.get_model_name(model_type)
        
        # Create cache key
        cache_key = f"{model_name}:{prompt[:50]}:{temperature}"
        
        # Check cache
        if cache_key in self.cache:
            logger.info(f"Cache hit for query with model {model_name}")
            return self.cache[cache_key]
        
        try:
            model = self.small_model if model_type == "small" else self.big_model
            
            # Format prompt for Ollama
            formatted_prompt = f"""
            {self._get_task_prompt()}
            
            {prompt}
            
            Please provide a detailed and creative response focusing on storytelling.
            """
            
            result = model.invoke(formatted_prompt)
            
            # Cache the result
            self.cache[cache_key] = result
            
            return result
        except Exception as e:
            logger.error(f"Error generating response with {model_name}: {str(e)}")
            return f"Error: {str(e)}"
    
    def _get_system_prompt(self):
        """Return system prompt for Ollama model."""
        return "You are an expert storyteller creating emotionally engaging narrative content. Focus on character development, dialogue, and emotional depth."
    
    def _get_task_prompt(self):
        """Return task-specific instructions."""
        return """As an expert radio dramatist and storyteller, create emotionally engaging narrative content with:
        - Distinctive character voices identifiable through dialogue alone
        - Sound-rich scenes that build a theater of the mind
        - Emotional moments that resonate deeply
        - Well-paced tension and release with engaging hooks
        - Effective use of narration and dialogue for audio storytelling"""

def query_models(prompt, model_type="small", use_chunks=False, temperature=0.7):
    """Query the appropriate model based on the task requirements."""
    handler = ModelFactory.get_handler()
    
    if use_chunks and len(prompt) > 3000:
        logger.info("Using chunking strategy for long prompt")
        return process_large_prompt(prompt, model_type, temperature)
    
    return handler.generate_response(prompt, model_type, temperature)

def process_large_prompt(prompt, model_type="big", temperature=0.7):
    """Process a large prompt by breaking it into chunks."""
    handler = ModelFactory.get_handler()
    
    # 1. Split the prompt into manageable chunks
    chunks = split_into_chunks(prompt, 2500)  # 2500 characters per chunk
    logger.info(f"Split prompt into {len(chunks)} chunks")
    
    # 2. Process each chunk with the small model for summarization
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        summary_prompt = f"""Summarize this portion of text while retaining all key emotional moments, character 
        development, and plot points that would be important for radio storytelling:

        {chunk}
        
        Focus on preserving dialogue patterns, emotional beats, and sound-rich scenes."""
        
        summary = handler.generate_response(summary_prompt, "small", temperature=0.5)
        chunk_summaries.append(summary)
        logger.info(f"Processed chunk {i+1}/{len(chunks)}")
    
    # 3. Combine the summaries
    combined_summary = "\n\n".join(chunk_summaries)
    
    # 4. Generate the final response with the preferred model
    final_prompt = f"""
    Use the following summarized context to generate your response for radio storytelling:
    
    {combined_summary}
    
    Based on this information, please respond to the original request, focusing on creating emotionally engaging content
    with distinctive character voices and sound-rich scenes that will captivate radio listeners.
    """
    
    return handler.generate_response(final_prompt, model_type, temperature=temperature)

def split_into_chunks(text, chunk_size):
    """Split text into chunks of roughly equal size at sentence boundaries."""
    chunks = []
    current_chunk = ""
    
    # Split by sentences (simplistic approach)
    sentences = text.replace(".", ".").replace("!", "!").replace("?", "?").split(".")
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        sentence = sentence.strip() + "."
        
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def query_with_memory(prompt, memory_data, model_type="small", temperature=0.7):
    """Generate a response using memory context."""
    handler = ModelFactory.get_handler()
    
    # Prepare memory context
    memory_context = format_memory_context(memory_data)
    
    # Create the full prompt with memory context
    full_prompt = f"""
    Context for radio storytelling:
    {memory_context}
    
    With this context in mind and focusing on creating emotionally engaging audio content with distinctive character voices,
    please respond to:
    {prompt}
    """
    
    logger.info("Querying with memory context")
    return handler.generate_response(full_prompt, model_type, temperature=temperature)

def format_memory_context(memory_data):
    """Format memory data into a coherent context for the model."""
    context_parts = []
    
    # Add basic story information
    if "story_title" in memory_data:
        context_parts.append(f"Story Title: {memory_data['story_title']}")
    
    if "concept" in memory_data:
        context_parts.append(f"Concept: {memory_data['concept']}")
    
    if "genre" in memory_data and "audience" in memory_data:
        context_parts.append(f"Genre: {memory_data['genre']}, Target Audience: {memory_data['audience']}")
    
    # Add character information with focus on voice and emotional traits
    if "characters" in memory_data and memory_data["characters"]:
        character_section = ["Characters (with voice traits and emotional backgrounds):"]
        for name, desc in memory_data["characters"].items():
            character_section.append(f"- {name}: {desc}")
        context_parts.append("\n".join(character_section))
    
    # Add previous episode summaries (limited by MAX_MEMORY_EPISODES)
    if "episode_summaries" in memory_data and memory_data["episode_summaries"]:
        summaries = memory_data["episode_summaries"]
        summaries_split = summaries.split("\n\n")
        
        # Take the most recent MAX_MEMORY_EPISODES
        recent_summaries = summaries_split[-MAX_MEMORY_EPISODES:] if len(summaries_split) > MAX_MEMORY_EPISODES else summaries_split
        
        context_parts.append("Previous Episode Emotional Arcs and Key Moments:")
        context_parts.append("\n".join(recent_summaries))
    
    # Add story bible highlights if available
    if "story_bible" in memory_data and memory_data["story_bible"]:
        # Extract just the important parts from the story bible
        bible_prompt = f"""
        Extract the most crucial elements from this story bible that would be necessary
        for creating emotionally engaging radio episodes with consistent character voices.
        Focus on:
        1. Character voice traits and emotional backgrounds
        2. Sound-rich scenes and atmospheric elements
        3. Emotional themes and character arcs
        4. Key relationship dynamics
        
        {memory_data["story_bible"][:2000]}  # Limit length for efficiency
        """
        
        bible_summary = query_models(bible_prompt, "small", temperature=0.5)
        context_parts.append("Story Bible Highlights for Radio Storytelling:")
        context_parts.append(bible_summary)
    
    return "\n\n".join(context_parts)