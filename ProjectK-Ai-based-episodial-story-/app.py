import streamlit as st
import os
from pipeline import generate_new_story, generate_new_episode, get_story_content
from config import USER_CONFIG, save_user_config, reload_config, get_model_provider

# Set page config at the very top of the script, before any other st commands
st.set_page_config(page_title="AI Episodic Storyteller", layout="wide")

def get_existing_stories():
    """Fetch existing story titles from saved story directories."""
    os.makedirs("stories", exist_ok=True)  # Ensure the stories directory exists
    stories = [d for d in os.listdir("stories") if os.path.isdir(os.path.join("stories", d))]
    return stories

def main():
    st.title("📚 AI-Powered Episodic Storytelling")
    st.sidebar.image("ai.jpg", width=150)
    
    # Configuration section in sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        configure_models()
        
        st.markdown("---")
        
        # App sections in sidebar
        # Initialize session state for navigation if not present
        if 'app_mode' not in st.session_state:
            st.session_state.app_mode = "Create Story"
        
        # Use session_state for the radio button to maintain state across reruns
        st.radio(
            "Choose Mode:",
            ["Create Story", "Continue Story", "Read Story"],
            key="app_mode"
        )
    
    # Now use the session state to determine which UI to show
    if st.session_state.app_mode == "Create Story":
        create_story_ui()
    elif st.session_state.app_mode == "Continue Story":
        continue_story_ui()
    elif st.session_state.app_mode == "Read Story":
        read_story_ui()

def configure_models():
    """UI for configuring model settings."""
    st.subheader("Model Settings")
    
    # Model provider selection
    current_provider = USER_CONFIG.get("model_provider", "ollama")
    selected_provider = st.selectbox(
        "Model Provider:", 
        options=["ollama", "openai"],
        index=0 if current_provider == "ollama" else 1,
        help="Select 'ollama' for offline local models or 'openai' for cloud-based models"
    )
    
    config_changed = False
    
    if selected_provider != current_provider:
        USER_CONFIG["model_provider"] = selected_provider
        config_changed = True
    
    # Provider-specific settings
    if selected_provider == "openai":
        current_api_key = USER_CONFIG.get("openai_api_key", "")
        api_key = st.text_input(
            "OpenAI API Key:", 
            value=current_api_key,
            type="password",
            help="Enter your OpenAI API key for GPT models"
        )
        
        if api_key != current_api_key:
            USER_CONFIG["openai_api_key"] = api_key
            config_changed = True
            
        # Model selection for OpenAI
        with st.expander("Advanced Model Settings", expanded=False):
            small_model = st.selectbox(
                "Small Model:",
                options=["gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
                index=0,
                help="Used for shorter tasks and summaries"
            )
            
            big_model = st.selectbox(
                "Big Model:",
                options=["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo-16k"],
                index=0,
                help="Used for generating full episodes and stories"
            )
            
            if small_model != USER_CONFIG.get("small_model_map", {}).get("openai"):
                if "small_model_map" not in USER_CONFIG:
                    USER_CONFIG["small_model_map"] = {}
                USER_CONFIG["small_model_map"]["openai"] = small_model
                config_changed = True
                
            if big_model != USER_CONFIG.get("big_model_map", {}).get("openai"):
                if "big_model_map" not in USER_CONFIG:
                    USER_CONFIG["big_model_map"] = {}
                USER_CONFIG["big_model_map"]["openai"] = big_model
                config_changed = True
    
    else:  # ollama
        current_url = USER_CONFIG.get("ollama_base_url", "http://localhost:11434")
        base_url = st.text_input(
            "Ollama API URL:",
            value=current_url,
            help="URL of your local Ollama server (default: http://localhost:11434)"
        )
        
        if base_url != current_url:
            USER_CONFIG["ollama_base_url"] = base_url
            config_changed = True
        
        # Model selection for Ollama
        with st.expander("Advanced Model Settings", expanded=False):
            # Get list of available models from the system
            small_model = st.selectbox(
                "Small Model:",
                options=["mistral", "llama2", "gemma", "phi"],
                index=0,
                help="Lighter model used for shorter tasks"
            )
            
            big_model = st.selectbox(
                "Big Model:",
                options=["llama3", "mixtral", "llama2:70b", "claude"],
                index=0,
                help="More powerful model used for generating stories"
            )
            
            if small_model != USER_CONFIG.get("small_model_map", {}).get("ollama"):
                if "small_model_map" not in USER_CONFIG:
                    USER_CONFIG["small_model_map"] = {}
                USER_CONFIG["small_model_map"]["ollama"] = small_model
                config_changed = True
                
            if big_model != USER_CONFIG.get("big_model_map", {}).get("ollama"):
                if "big_model_map" not in USER_CONFIG:
                    USER_CONFIG["big_model_map"] = {}
                USER_CONFIG["big_model_map"]["ollama"] = big_model
                config_changed = True
    
    # Save configuration if changes were made
    if config_changed:
        if save_user_config(USER_CONFIG):
            # Force configuration reload
            reload_config()
            st.success(f"✅ Settings saved! Now using: {get_model_provider()} provider.")
        else:
            st.error("❌ Failed to save settings.")

def create_story_ui():
    st.header("📝 Create a New Story")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Story Parameters")
        brief_text = st.text_area(
            "Enter your story concept:",
            height=150,
            placeholder="e.g., A detective with the ability to see 5 minutes into the future tries to solve a case where this power becomes unreliable..."
        )
        
        genre = st.selectbox("Select genre:", 
            ["Science Fiction", "Fantasy", "Mystery", "Romance", "Thriller", "Comedy", "Drama", "Adventure"]
        )
        
        target_audience = st.selectbox("Target audience:", 
            ["Children", "Young Adult", "Adult", "All Ages"]
        )
    
    with col2:
        st.subheader("Episode Settings")
        num_episodes = st.slider("Initial episodes:", 1, 5, 2)
        episode_length = st.select_slider(
            "Episode length:",
            options=["Short", "Medium", "Long"],
            value="Medium"
        )
        
        # Advanced settings (collapsible)
        with st.expander("Advanced Settings"):
            model_quality = st.select_slider(
                "Model quality:",
                options=["Standard", "Premium"],
                value="Standard",
                help="Premium uses a more advanced model but may take longer"
            )
            
            creativity = st.slider(
                "Creativity level:",
                0.0, 1.0, 0.7,
                help="Higher values mean more creative but possibly less coherent stories"
            )
    
    if st.button("🪄 Generate Story", use_container_width=True):
        if brief_text.strip():
            # Check if using OpenAI and API key is missing
            if get_model_provider() == "openai" and not USER_CONFIG.get("openai_api_key"):
                st.error("⚠️ OpenAI API Key is required when using OpenAI models. Please add it in the configuration section.")
                return
                
            with st.spinner("Creating your story... this may take a minute"):
                try:
                    title = generate_new_story(
                        brief_text, 
                        num_episodes,
                        genre=genre,
                        audience=target_audience,
                        length=episode_length,
                        model_quality="big" if model_quality == "Premium" else "small",
                        temperature=creativity
                    )
                    
                    if title:
                        st.success(f"✅ Story '{title}' created with {num_episodes} episodes!")
                        st.session_state.current_story = title
                        st.balloons()
                        
                        # Show button to view the story
                        if st.button("Read Your Story Now"):
                            # Set session state to navigate to Read Story mode
                            st.session_state.app_mode = "Read Story"
                            st.experimental_rerun()
                    else:
                        st.error("⚠️ Story generation failed. Please try again.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter a story concept to continue.")

def continue_story_ui():
    st.header("📖 Continue an Existing Story")
    
    story_titles = get_existing_stories()
    
    if not story_titles:
        st.warning("No existing stories found. Create a new story first.")
        return
    
    selected_story = st.selectbox("Select a story to continue:", story_titles)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        character_input = st.text_area(
            "New characters to introduce (optional):",
            placeholder="e.g., Marina, a hacker with a mysterious past who knows more than she lets on..."
        )
        
        plot_direction = st.text_area(
            "Plot direction (optional):",
            placeholder="e.g., Introduce a twist where the main character discovers a betrayal..."
        )
    
    with col2:
        model_quality = st.select_slider(
            "Model quality:",
            options=["Standard", "Premium"],
            value="Premium"
        )
        
        with st.expander("Advanced Settings"):
            maintain_tone = st.checkbox("Maintain existing tone", value=True)
            creativity = st.slider("Creativity level:", 0.0, 1.0, 0.7)
    
    if st.button("🚀 Generate Next Episode", use_container_width=True):
        # Check if using OpenAI and API key is missing
        if get_model_provider() == "openai" and not USER_CONFIG.get("openai_api_key"):
            st.error("⚠️ OpenAI API Key is required when using OpenAI models. Please add it in the configuration section.")
            return
            
        with st.spinner("Crafting the next chapter of your story..."):
            try:
                new_episode = generate_new_episode(
                    selected_story, 
                    character_input,
                    plot_direction=plot_direction,
                    model_quality="big" if model_quality == "Premium" else "small",
                    maintain_tone=maintain_tone,
                    temperature=creativity
                )
                
                if new_episode:
                    st.success("✅ New episode added successfully!")
                    
                    with st.expander("📜 Preview New Episode", expanded=True):
                        st.markdown(new_episode)
                    
                    if st.button("Read Full Story"):
                        st.session_state.current_story = selected_story
                        # Navigate to Read Story mode
                        st.session_state.app_mode = "Read Story"
                        st.experimental_rerun()
                else:
                    st.error("⚠️ Episode generation failed. Please try again.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def read_story_ui():
    st.header("📚 Read Your Stories")
    
    story_titles = get_existing_stories()
    
    if not story_titles:
        st.warning("No stories found. Create a new story first.")
        return
    
    # If coming from another page with a selected story
    current_story = st.session_state.get("current_story", story_titles[0])
    selected_story = st.selectbox("Select a story to read:", story_titles, index=story_titles.index(current_story) if current_story in story_titles else 0)
    
    story_content = get_story_content(selected_story)
    if story_content:
        # Parse episodes
        episodes = story_content.split("\n\nEpisode ")
        title_section = episodes[0]
        actual_episodes = ["Episode " + ep for ep in episodes[1:]]
        
        # Display title
        st.subheader(selected_story.replace("_", " ").title())
        
        # Episode selection
        if len(actual_episodes) > 1:
            episode_options = ["All Episodes"] + [f"Episode {i+1}" for i in range(len(actual_episodes))]
            selected_episode = st.selectbox("Choose episode:", episode_options)
        else:
            selected_episode = "All Episodes"
        
        # Display episodes
        if selected_episode == "All Episodes":
            for i, episode in enumerate(actual_episodes):
                with st.expander(f"Episode {i+1}", expanded=i==0):
                    st.markdown(episode)
        else:
            episode_index = int(selected_episode.split(" ")[1]) - 1
            st.markdown(actual_episodes[episode_index])
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✏️ Continue This Story"):
                st.session_state.current_story = selected_story
                # Navigate to Continue Story mode
                st.session_state.app_mode = "Continue Story"
                st.experimental_rerun()
        with col2:
            if st.button("📊 Story Analytics"):
                # Basic analytics could be added here
                st.info("Story analytics feature coming soon!")
    else:
        st.error("Could not load story content.")

if __name__ == "__main__":
    main()