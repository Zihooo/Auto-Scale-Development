"""
Configuration module for auto_scale_development package.

This module handles API key management
"""

import os
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv

# Import API clients
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from together import Together
except ImportError:
    Together = None

# Model configuration mapping
MODEL_CONFIG = {
    # OpenAI models
    "o4-mini-2025-04-16": {"provider": "openai", "api_key_env": "OPENAI_API_KEY"},
    "chatgpt-4o-latest": {"provider": "openai", "api_key_env": "OPENAI_API_KEY"}, 
    "gpt-4.1-2025-04-14": {"provider": "openai", "api_key_env": "OPENAI_API_KEY"},
    "o3-2025-04-16": {"provider": "openai", "api_key_env": "OPENAI_API_KEY"},
    
    # Anthropic models
    "claude-opus-4-20250514": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    "claude-sonnet-4-20250514": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    "claude-3-7-sonnet-20250219": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    "claude-3-5-haiku-20241022": {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    
    # Google models
    "gemini-2.5-pro": {"provider": "google", "api_key_env": "GOOGLE_API_KEY"},
    "gemini-2.5-flash": {"provider": "google", "api_key_env": "GOOGLE_API_KEY"},
    
    # Together AI models
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"provider": "together", "api_key_env": "TOGETHER_API_KEY"},
    "Qwen/QwQ-32B": {"provider": "together", "api_key_env": "TOGETHER_API_KEY"},
    "mistralai/Mistral-7B-Instruct-v0.2": {"provider": "together", "api_key_env": "TOGETHER_API_KEY"},
}


def get_api_key(model_name: str) -> Optional[str]:
    """
    Get API key from .env file based on model name.
    
    Args:
        model_name (str): Name of the model to get API key for.
    
    Returns:
        Optional[str]: The API key if found, None otherwise.
    
    Example:
        >>> api_key = get_api_key("gpt-4.1-2025-04-14")
        >>> print(f"API Key found: {api_key is not None}")
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the model configuration
    model_config = MODEL_CONFIG.get(model_name)
    
    if model_config is None:
        print(f"Warning: Model '{model_name}' not found in model mapping")
        print(f"Available models: {list(MODEL_CONFIG.keys())}")
        return None
    
    # Get API key from environment variables
    api_key = os.getenv(model_config["api_key_env"])
    
    if api_key is None:
        print(f"Warning: {model_config['api_key_env']} not found in .env file")
        return None
    
    return api_key


def get_model_provider(model_name: str) -> Optional[str]:
    """
    Get the provider for a given model.
    
    Args:
        model_name (str): Name of the model.
    
    Returns:
        Optional[str]: The provider name if found, None otherwise.
    
    Example:
        >>> provider = get_model_provider("gpt-4.1-2025-04-14")
        >>> print(provider)  # "openai"
    """
    model_config = MODEL_CONFIG.get(model_name)
    return model_config["provider"] if model_config else None


def call_llm_api(
    model_name: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 1.0,
    top_p: float = 0.8,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None) -> str:
    """
    Unified wrapper function to call different LLM APIs.
    
    Args:
        model_name (str): Name of the model to use
        user_prompt (str): The user's prompt/message
        system_prompt (Optional[str]): System prompt/instruction. Defaults to None.
        temperature (float): Controls randomness (0.0 to 2.0). Defaults to 1.0.
        top_p (float): Controls diversity via nucleus sampling (0.0 to 1.0). Defaults to 0.8.
        max_tokens (Optional[int]): Maximum number of tokens to generate. Defaults to None.
        api_key (Optional[str]): API key to use. If None, will try to get from .env file.
    
    Returns:
        str: The generated response text
    
    Raises:
        ValueError: If model is not supported or required packages are not installed
        Exception: If API call fails
    
    Example:
        >>> response = call_llm_api(
        ...     model_name="gpt-4.1-2025-04-14",
        ...     system_prompt="You are a helpful assistant.",
        ...     user_prompt="How does a CPU work?",
        ...     temperature=0.7,
        ...     top_p=0.9
        ... )
        >>> print(response)
    """
    # Get model configuration
    model_config = MODEL_CONFIG.get(model_name)
    if model_config is None:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Get API key if not provided
    if api_key is None:
        api_key = get_api_key(model_name)
        if api_key is None:
            raise ValueError(f"API key not found for model: {model_name}")
    
    try:
        provider = model_config["provider"]
        if provider == "openai":
            return _call_openai_api(
                model_name, user_prompt, system_prompt, 
                temperature, top_p, max_tokens, api_key
            )
        elif provider == "anthropic":
            return _call_anthropic_api(
                model_name, user_prompt, system_prompt, 
                temperature, top_p, max_tokens, api_key
            )
        elif provider == "google":
            return _call_google_api(
                model_name, user_prompt, system_prompt, 
                temperature, top_p, max_tokens, api_key
            )
        elif provider == "together":
            return _call_together_api(
                model_name, user_prompt, system_prompt, 
                temperature, top_p, max_tokens, api_key
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    except Exception as e:
        raise Exception(f"API call failed for {model_name}: {str(e)}")


def _call_openai_api(
    model_name: str,
    user_prompt: str,
    system_prompt: Optional[str],
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    api_key: str) -> str:
    """Call OpenAI API."""
    if OpenAI is None:
        raise ValueError("OpenAI package not installed. Install with: pip install openai")
    
    client = OpenAI(api_key=api_key)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    
    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "messages": messages
    }
    
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    
    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content


def _call_anthropic_api(
    model_name: str,
    user_prompt: str,
    system_prompt: Optional[str],
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    api_key: str) -> str:
    """Call Anthropic API."""
    if anthropic is None:
        raise ValueError("Anthropic package not installed. Install with: pip install anthropic")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "messages": [{"role": "user", "content": user_prompt}]
    }
    
    if system_prompt:
        kwargs["system"] = system_prompt
    
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = 8192  # Default for Anthropic
    
    completion = client.messages.create(**kwargs)
    return completion.content[0].text.strip()


def _call_google_api(
    model_name: str,
    user_prompt: str,
    system_prompt: Optional[str],
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    api_key: str) -> str:
    """Call Google API."""
    if genai is None:
        raise ValueError("Google Generative AI package not installed. Install with: pip install google-generativeai")
    
    genai.configure(api_key=api_key)
    
    config = {
        "temperature": temperature,
        "top_p": top_p,
    }
    
    if max_tokens:
        config["max_output_tokens"] = max_tokens
    
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=config,
        system_instruction=system_prompt if system_prompt else ""
    )
    
    response = model.generate_content(user_prompt)
    return response.text


def _call_together_api(
    model_name: str,
    user_prompt: str,
    system_prompt: Optional[str],
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    api_key: str) -> str:
    """Call Together AI API."""
    if Together is None:
        raise ValueError("Together AI package not installed. Install with: pip install together")
    
    client = Together(api_key=api_key)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    
    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "messages": messages
    }
    
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    
    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content

