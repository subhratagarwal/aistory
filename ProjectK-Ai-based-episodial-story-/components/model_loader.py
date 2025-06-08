import ollama

def load_model(model_name):
    """Ensure model is downloaded and ready to use."""
    ollama.pull(model_name)  # Ensures the model is available
    return model_name  # Just return the model name, as `ollama.chat()` uses it directly