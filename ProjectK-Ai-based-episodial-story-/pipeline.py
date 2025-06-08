import os
import json
import logging
from datetime import datetime
from query_handler import query_models, query_with_memory
import re

# Define directories
STORIES_DIR = "stories"
LOG_FILE = "story_pipeline.log"

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

def sanitize_title(title):
    """Sanitize story title for use in file names."""
    title = " ".join(title.split()[:5])  # Limit to 5 words
    return "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).replace(" ", "_").strip()

def save_to_file(file_path, content, mode="w"):
    """Save content to a file safely."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode, encoding="utf-8") as f:
            f.writelines(content)
        logger.info(f"Saved content to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save file {file_path}: {e}")
        raise RuntimeError(f"File save error: {e}")

def load_from_file(file_path):
    """Load content from a file safely."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None

def load_metadata(story_path):
    """Load story metadata from JSON file."""
    metadata_path = os.path.join(story_path, "metadata.json")
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Metadata file not found: {metadata_path}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in metadata file: {metadata_path}")
        return {}
    except Exception as e:
        logger.error(f"Error reading metadata file {metadata_path}: {e}")
        return {}

def save_metadata(story_path, metadata):
    """Save story metadata to JSON file."""
    metadata_path = os.path.join(story_path, "metadata.json")
    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save metadata file {metadata_path}: {e}")
        return False

def extract_characters(episode_content):
    """Extract character names from episode content."""
    # Use a more advanced prompt to extract character information
    character_prompt = f"""
    Extract all character information from this episode text that would help voice actors perform these roles.
    For each character include:
    - Name
    - Voice characteristics (tone, accent, speech patterns)
    - Emotional state and personality traits
    - Key relationships to other characters

    Format as JSON with name as key and detailed description as value.
    Only include named characters with actual speaking roles or significant mentions:
    
    {episode_content}
    """
    
    character_data = query_models(character_prompt, model_type="small")
    
    # Try to parse as JSON, fall back to regex if that fails
    try:
        char_dict = json.loads(character_data)
        return char_dict
    except json.JSONDecodeError:
        # Fallback to regex pattern matching
        logger.warning("JSON parsing failed for character extraction, using regex fallback")
        pattern = r"(?:\"|\')([A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*)?)(?:\"|\'):\s*(?:\"|\')([^\"\']+)(?:\"|\')"
        matches = re.findall(pattern, character_data)
        return {name: desc for name, desc in matches}

def generate_new_story(brief_text, num_episodes, genre="Adventure", audience="Adult", length="Medium", model_quality="small", temperature=0.7):
    """Generate a new story based on a brief idea with multiple episodes."""
    if not brief_text.strip():
        logger.error("Empty story brief provided")
        return None

    logger.info(f"Generating story for idea: {brief_text[:50]}...")

    # Generate a unique story title
    title_prompt = f"""
    Generate a catchy, memorable story title (3-5 words) that would work well for a radio drama based on this concept:
    
    Concept: {brief_text}
    Genre: {genre}
    Audience: {audience}
    
    Create a title that's ear-catching, easy to remember when heard (not read), and intriguing.
    Title:
    """
    
    raw_title = query_models(title_prompt, model_type="small", temperature=temperature)
    story_title = sanitize_title(raw_title)

    if not story_title:
        logger.error("Could not generate a valid story title")
        return None

    story_path = os.path.join(STORIES_DIR, story_title)
    os.makedirs(story_path, exist_ok=True)

    logger.info(f"Created directory for story: {story_path}")

    # Initialize metadata
    metadata = {
        "title": raw_title.strip(),
        "created_at": datetime.now().isoformat(),
        "concept": brief_text,
        "genre": genre,
        "audience": audience,
        "episode_length": length,
        "episodes": [],
        "characters": {}
    }

    # Create story bible/outline first
    bible_prompt = f"""
    Create a comprehensive story bible for a radio-friendly, emotionally engaging multi-episode story:

    Title: {raw_title}
    Concept: {brief_text}
    Genre: {genre}
    Target Audience: {audience}

    Your story bible should create a vivid audio experience that captivates radio listeners by:

    1. Main plot arc summary with emotional high points and moments of tension that will grip listeners
    2. 3-5 primary characters with:
       - Distinctive names that are easy to distinguish when heard (not read)
       - Rich emotional backgrounds and motivations
       - Unique vocal traits or speech patterns to help listeners identify them
       - Clear relationships and dynamics between them
    3. Vivid settings or locations with atmospheric details that listeners can imagine
    4. Sound-rich scenes and scenarios (what sounds would enhance each scene?)
    5. Themes and emotional motifs to develop across episodes
    6. Basic episode structure for {num_episodes} episodes with cliffhangers and emotional hooks
    7. Opportunities for audio drama techniques (inner monologues, ambient sounds, etc.)

    Remember this is for audio storytelling, so dialogue and sound-rich scenes are more important than visual descriptions.
    """
    
    story_bible = query_models(bible_prompt, model_type="big", temperature=temperature * 0.9)
    save_to_file(os.path.join(story_path, "story_bible.txt"), story_bible)
    
    # Extract and save character information to metadata
    character_prompt = f"""
    Extract the character information from this story bible.
    For each character, provide details that would help voice actors bring them to life:
    - Name
    - Voice characteristics (tone, accent, speech patterns)
    - Emotional background and personality
    - Key relationships
    
    Return as JSON with character names as keys and descriptions as values:
    
    {story_bible}
    """
    
    character_data = query_models(character_prompt, model_type="small")
    try:
        # Try to parse as JSON
        char_dict = json.loads(character_data)
        metadata["characters"] = char_dict
    except json.JSONDecodeError:
        logger.warning("JSON parsing failed for character extraction from story bible")
        # Use regex as fallback
        pattern = r"(?:\"|\')([A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*)?)(?:\"|\'):\s*(?:\"|\')([^\"\']+)(?:\"|\')"
        matches = re.findall(pattern, character_data)
        metadata["characters"] = {name: desc for name, desc in matches}

    save_metadata(story_path, metadata)

    # Generate episodes
    all_content = [f"# {raw_title.strip()}\n\n"]
    episode_summaries = []

    for ep in range(1, num_episodes + 1):
        try:
            # Adjust prompt complexity based on episode number
            if ep == 1:
                # First episode establishes everything
                episode_prompt = f"""
                Write Episode {ep} for "{raw_title}" designed specifically for radio storytelling.

                Story Concept: {brief_text}
                Genre: {genre}
                Audience: {audience}

                Following radio drama best practices:
                1. Begin with a powerful hook or sound-rich scene that immediately captures attention
                2. Introduce characters with distinctive voices and speech patterns that listeners can easily identify
                3. Create emotional moments that resonate with listeners
                4. Balance narration with engaging dialogue
                5. Include moments where listeners can hear what characters are thinking or feeling
                6. End with a hook that makes listeners eager for the next episode

                Use the story bible to introduce main characters and establish the premise.
                Length: {length} (about {length.lower()} length for reading aloud)

                Start with "Episode {ep}:" and then the content.
                """
            else:
                # Later episodes build on previous ones
                previous_summaries = "\n".join(episode_summaries)
                episode_prompt = f"""
                Write Episode {ep} for "{raw_title}" designed specifically for radio storytelling.

                Previous episode summaries:
                {previous_summaries}

                Following radio drama best practices:
                1. Begin with a brief recap of relevant previous events
                2. Use distinctive dialogue and speech patterns for each character
                3. Create emotional moments and conflicts that listeners can feel
                4. Include sound-rich scenes that create a theater of the mind
                5. Maintain a pace that keeps listeners engaged
                6. End with a compelling hook for the next episode

                Continue developing the story and characters with emotional depth.
                Maintain consistent character voices and relationship dynamics.
                Length: {length} (about {length.lower()} length for reading aloud)

                Start with "Episode {ep}:" and then the content.
                """

            episode_content = query_models(
                episode_prompt, 
                model_type=model_quality, 
                use_chunks=True,
                temperature=temperature
            )

            if not episode_content:
                logger.warning(f"Episode {ep} generation failed")
                continue

            # Clean up formatting if needed
            if not episode_content.startswith(f"Episode {ep}:"):
                episode_content = f"Episode {ep}:\n{episode_content}"
                
            all_content.append(episode_content)

            # Extract characters from this episode
            episode_characters = extract_characters(episode_content)
            
            # Update metadata characters with any new ones
            for char_name, char_desc in episode_characters.items():
                if char_name not in metadata["characters"]:
                    metadata["characters"][char_name] = char_desc

            # Generate episode summary for continuity
            summary_prompt = f"""
            Summarize the key plot points, character development, emotional moments, and important events from this episode:
            
            {episode_content}
            
            Focus on emotional beats and character growth that will be important for continuity.
            Keep the summary concise but include all plot-critical information.
            """
            
            episode_summary = query_models(summary_prompt, model_type="small")
            episode_summaries.append(f"Episode {ep} Summary: {episode_summary}")
            
            # Add episode to metadata
            metadata["episodes"].append({
                "number": ep,
                "title": f"Episode {ep}",
                "created_at": datetime.now().isoformat(),
                "summary": episode_summary,
                "characters": list(episode_characters.keys())
            })
            
            # Update metadata after each episode
            save_metadata(story_path, metadata)

        except Exception as e:
            logger.error(f"Error generating Episode {ep}: {e}")

    # Save the complete story
    save_to_file(os.path.join(story_path, "story.txt"), "\n\n".join(all_content))
    save_to_file(os.path.join(story_path, "story_summaries.txt"), "\n\n".join(episode_summaries))

    logger.info(f"Story '{story_title}' generated successfully with {num_episodes} episodes")
    return story_title

def generate_new_episode(story_title, character_input="", plot_direction="", model_quality="big", maintain_tone=True, temperature=0.7):
    """Generate a new episode for an existing story, considering additional characters and plot directions."""
    story_path = os.path.join(STORIES_DIR, story_title)
    story_file = os.path.join(story_path, "story.txt")
    summaries_file = os.path.join(story_path, "story_summaries.txt")
    bible_file = os.path.join(story_path, "story_bible.txt")

    if not os.path.exists(story_file):
        logger.error(f"Story file missing for: {story_title}")
        return None

    # Load metadata
    metadata = load_metadata(story_path)
    current_episode_count = len(metadata.get("episodes", []))
    next_episode_number = current_episode_count + 1
    
    # Load story bible and previous content
    story_bible = load_from_file(bible_file) or ""
    summaries = load_from_file(summaries_file) or ""
    
    # Prepare memory for the model
    memory_data = {
        "story_title": metadata.get("title", story_title),
        "concept": metadata.get("concept", ""),
        "genre": metadata.get("genre", ""),
        "audience": metadata.get("audience", ""),
        "characters": metadata.get("characters", {}),
        "episode_summaries": summaries,
        "story_bible": story_bible
    }

    # Build context for the model
    context = query_with_memory(
        f"Prepare context for episode {next_episode_number} of '{story_title}', focusing on emotional continuity and character voices",
        memory_data,
        model_type="small"
    )
    
    # Create the episode prompt
    tone_guidance = "Maintain the same tone, emotional rhythm, and writing style as previous episodes." if maintain_tone else ""
    
    episode_prompt = f"""
    Write Episode {next_episode_number} for "{metadata.get('title', story_title)}" designed specifically for radio storytelling.

    Context (previous episodes and characters):
    {context}

    {f'New characters to introduce (with distinctive voices): {character_input}' if character_input else ''}
    {f'Plot direction with emotional moments: {plot_direction}' if plot_direction else ''}
    {tone_guidance}

    Radio storytelling guidelines:
    1. Use rich, evocative dialogue that reveals character emotions
    2. Create scenes with atmospheric sounds and emotional weight
    3. Balance narration with character interactions
    4. Maintain a rhythm that keeps listeners engaged
    5. Include moments of tension, release, revelation, and emotional impact
    6. End with a hook that makes listeners want more

    Make sure listeners can feel the emotional journey of each character.
    Maintain continuity with previous episodes.
    Length: {metadata.get('episode_length', 'Medium')}

    Start with "Episode {next_episode_number}:" and then the content.
    """

    new_episode = query_models(
        episode_prompt, 
        model_type=model_quality, 
        use_chunks=True,
        temperature=temperature
    )

    if not new_episode:
        logger.error(f"Failed to generate episode {next_episode_number} for '{story_title}'")
        return None

    # Clean up formatting if needed
    if not new_episode.startswith(f"Episode {next_episode_number}:"):
        new_episode = f"Episode {next_episode_number}:\n{new_episode}"
    
    # Update the story file
    story_content = load_from_file(story_file)
    updated_story = f"{story_content}\n\n{new_episode}" if story_content else new_episode
    save_to_file(story_file, updated_story)
    
    # Extract new characters
    episode_characters = extract_characters(new_episode)
    
    # Update metadata characters with any new ones
    for char_name, char_desc in episode_characters.items():
        if char_name not in metadata["characters"]:
            metadata["characters"][char_name] = char_desc
    
    # Generate episode summary
    summary_prompt = f"""
    Summarize the key plot points, character development, emotional moments, and important events from this episode:
    
    {new_episode}
    
    Focus on the emotional journey of characters and moments that will resonate with listeners.
    Keep the summary concise but include all plot-critical information.
    """
    
    episode_summary = query_models(summary_prompt, model_type="small")
    
    # Update episode summaries file
    updated_summaries = f"{summaries}\n\nEpisode {next_episode_number} Summary: {episode_summary}" if summaries else f"Episode {next_episode_number} Summary: {episode_summary}"
    save_to_file(summaries_file, updated_summaries)
    
    # Add episode to metadata
    metadata["episodes"].append({
        "number": next_episode_number,
        "title": f"Episode {next_episode_number}",
        "created_at": datetime.now().isoformat(),
        "summary": episode_summary,
        "characters": list(episode_characters.keys())
    })
    
    # Update metadata
    save_metadata(story_path, metadata)

    logger.info(f"New episode {next_episode_number} added to '{story_title}'")
    return new_episode

def get_story_content(story_title):
    """Get the full content of a story."""
    story_path = os.path.join(STORIES_DIR, story_title)
    story_file = os.path.join(story_path, "story.txt")
    
    return load_from_file(story_file)