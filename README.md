# Auto Scale Development

A Python package for automated scale development and analysis.

![demo](/auto_scale_development/demo.jpg)

## Architecture

```
auto_scale_development/
├── __init__.py              # Package initialization and public API
├── config.py                # API configuration and LLM provider management
├── core_functions.py        # Main scale development functions
├── helper_functions.py      # Internal utility functions
├── prompts.py               # LLM prompt templates
└── demo.jpg                 # Demo image
```


## item_generation

```python
items = item_generation(
    construct=construct,
    definition=definition,
    dimensions=dimensions,
    examples=examples,
    num_items=5,  # Generate 5 items per dimension (15 total)
    model_name="gpt-4.1-2025-04-14",
    temperature=1.0,
    top_p=0.8
    api_key=None  # Will use default API key
)
```

## item_reduction

```python
filtered_items = item_reduction(items, similarity_threshold=0.8, verbose=True)
```

## content_validation

```python
validated_items = content_validation(
    items_dict=filtered_items,
    definitions=validation_definitions,
    scale_points=7,
    models=["gpt-4.1-2025-04-14", "chatgpt-4o-latest"],
    runs_per_model=2,  # Run each model multiple times for reliability
    top_n_per_dimension=5,  # Select top 5 items per dimension
    api_key=None  # Will use default API key
)
```

## get_items

```python
get_items(validated_items)   # Return a dictionary of items
get_items(validated_items['top_items'])   # Return a dictionary of top items
```


## export_items_to_excel

```python
excel_file = export_items_to_excel(items, "items.xlsx")
```

## export_items_to_json

```python
json_file = export_items_to_json(items, "items.json")
```


## .env file

```
# OpenAI API Key
OPENAI_API_KEY=
# o4-mini-2025-04-16
# chatgpt-4o-latest
# gpt-4.1-2025-04-14
# o3-2025-04-16


# Anthropic API Key
ANTHROPIC_API_KEY=
# claude-opus-4-20250514
# claude-sonnet-4-20250514
# claude-3-7-sonnet-20250219
# claude-3-5-haiku-20241022


# Google API Key
GOOGLE_API_KEY=
# gemini-2.5-pro
# gemini-2.5-flash


# Together AI API key
TOGETHER_API_KEY=
# meta-llama/Llama-3.3-70B-Instruct-Turbo
# Qwen/QwQ-32B
# mistralai/Mistral-7B-Instruct-v0.2
```
