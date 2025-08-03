# Auto Scale Development

A Python package for automated scale development and analysis.

![demo](/auto_scale_development/demo.jpg)

## Installation

### Install package
```bash
pip install auto-scale-development
```

### Install from GitHub
```bash
pip install git+https://github.com/your-username/Auto-Scale-Development.git
```

### Set API_KE configuration (.env file)

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



## Architecture

```
Auto-Scale-Development/
├── auto_scale_development/           # Main package directory
│   ├── __init__.py                  # Package initialization and public API
│   ├── config.py                    # API key management and LLM provider routing
│   ├── core_functions.py            # Main functions (item_generation, validation, etc.)
│   ├── helper_functions.py          # Utility functions for data processing
│   ├── prompts.py                   # System and user prompts for LLM interactions
│   └── demo.jpg                     # Demo image
├── functions_example.py             # Complete usage example
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project configuration and metadata
├── setup.py                         # Package setup script
├── MANIFEST.in                      # Package manifest
├── LICENSE                          # MIT license
└── README.md                        # Documentation
```



## item_generation

```python
items = item_generation(
    construct=construct,
    definition=definition,
    dimensions=dimensions,
    examples=examples,
    num_items=40,  # Generate 40 items per dimension (120 total)
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
    top_n_per_dimension=10,  # Select top 10 items per dimension
    api_key=None  # Will use default API key
)

validated_items
validated_items['top_items']
```


## statement_pair

```python
pairs_dict = statement_pair(
    items_input=validated_items,  # Can use output from any function
    output_file="statement_pairs.xlsx",  # Automatically export to Excel
    balance_data=True,  # Balance label=0 and label=1 counts  
    random_seed=42  # For reproducible results
)
```

## fine_tune

```python
model = fine_tune(
    pairs_input=pairs_dict,
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Base model
    output_path="./fine_tuned_model", 
    train_test_split=0.8,  # 80% train, 20% test
    num_train_epochs=5,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    loss_function="MultipleNegativesRankingLoss",
    random_seed=42
)
```

### Using the Fine-tuned Model

```python
# Load and use your fine-tuned model
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("./fine_tuned_model")

# Calculate embeddings
statements = ["I worry about others' intentions", "I trust people easily"]
embeddings = model.encode(statements)

# Calculate similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f"Similarity: {similarity:.4f}")
```

## EFA_item_selection

```python
efa_results = EFA_item_selection(
    items_input=validated_items["top_items"],  # Direct use of top_items format
    model_path="./fine_tuned_model",  # Path to the fine-tuned model
    n_factors=3,  # Extract 3 factors (Cognitive, Behavioral, Affective)
    items_per_factor=5,  # Select top 5 items per factor
    rotation="oblimin",  # Varimax rotation for clearer factor structure
    random_seed=42
)

efa_results["selected_items"]
efa_results["factor_loadings_file"]
efa_results["plot_file"]
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



