from auto_scale_development import item_generation, export_items_to_excel, export_items_to_json, item_reduction, content_validation, statement_pair, fine_tune, EFA_item_selection

# item_generation
# Define the construct and its dimensions
construct = "Interpersonal Distrust"
definition = "An expectation of harmful, hostile, or other negative outcomes in interactions with another person, which may be based on past experiences, observations, or general beliefs about human nature."

dimensions = {
    "Cognitive": "Rational beliefs or expectations about another party's untrustworthiness, including suspicions about their motives, intentions, or reliability.",
    "Behavioral": "The unwillingness or avoidance of future interactions or risk-taking actions with the distrusted person, including protective behaviors.",
    "Affective": "Negative emotions directed at the distrusted person, such as anxiety, fear, anger, or discomfort."
}

# Optional: Provide examples for each dimension
examples = {
    "Cognitive": [
        "This person would behave in a deceptive and fraudulent way.",
        "I am suspicious of the way this person will act in the future."
    ],
    "Behavioral": [
        "I find it necessary to be cautious with this person.",
        "I will protect myself from being taken advantage of by this person."
    ],
    "Affective": [
        "I feel tense when I am with this person.",
        "I experience anxiety when interacting with this person."
    ]
}

## Sample call to item_generation function
# Generate 5 items per dimension (15 total items)
items = item_generation(
    construct=construct,
    definition=definition,
    dimensions=dimensions,
    examples=examples,
    num_items=40,
    model_name="chatgpt-4o-latest",  # Using a faster model for testing
    temperature=1.0,
    top_p=0.8
)

print("Generated Items:")
print(items)


# item_reduction
filtered_items = item_reduction(items, similarity_threshold=0.8, verbose=True)
print(f"\nFiltered Items:")
print(f"Original items: {filtered_items['metadata']['original_total_items']}")
print(f"Items after filtering: {filtered_items['metadata']['total_items']}")
print(f"Items removed: {filtered_items['metadata']['items_removed']}")

print("Filtered Items:")
print(filtered_items)

# ## Export to Excel
# print("\n" + "=" * 60)
# print("Exporting filtered items to Excel...")
# excel_file = export_items_to_excel(filtered_items, "interpersonal_distrust_items_filtered.xlsx")
# print(f"Excel file created: {excel_file}")

# ## Export to JSON
# print("\n" + "=" * 60)
# print("Exporting filtered items to JSON...")
# json_file = export_items_to_json(filtered_items, "interpersonal_distrust_items_filtered.json")
# print(f"JSON file created: {json_file}")


# content_validation

validation_definitions = {
    "Cognitive": "Rational beliefs or expectations about another party's untrustworthiness, including suspicions about their motives, intentions, or reliability.",
    "Behavioral": "The unwillingness or avoidance of future interactions or risk-taking actions with the distrusted person, including protective behaviors.",
    "Affective": "Negative emotions directed at the distrusted person, such as anxiety, fear, anger, or discomfort."
}

validated_items = content_validation(
    items_dict=filtered_items,
    definitions=validation_definitions,
    scale_points=7,
    models=["chatgpt-4o-latest", "gpt-4.1-2025-04-14"], 
    runs_per_model=2,
    top_n_per_dimension=10,
    api_key=None  # Will use environment variable
)

print("Validated Items:")
print(validated_items)


# statement_pair
print("\n" + "=" * 60)
print("Creating statement pairs...")

# Create statement pairs from validated items
pairs_dict = statement_pair(
    items_input=validated_items,
    output_file="statement_pairs.xlsx",  # Automatically export to Excel
    balance_data=True,  # Balance label=0 and label=1 counts
    random_seed=42  # For reproducible results
)


print("Statement pairs have been exported to 'statement_pairs.xlsx'")


# fine_tune
print("\n" + "=" * 60)
print("Fine-tuning sentence transformer model...")

# Fine-tune model using the statement pairs
fine_tuned_model = fine_tune(
    pairs_input=pairs_dict,  # Can also use "statement_pairs.xlsx"
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    output_path="./fine_tuned_model",
    train_test_split=0.8,  # 80% for training, 20% for testing
    num_train_epochs=3,  # Reduced for faster testing
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # Smaller batch size for testing
    warmup_steps=50,
    loss_function="MultipleNegativesRankingLoss",  # or "CosineSimilarityLoss"
    random_seed=42
)

print(f"Fine-tuning completed! Model type: {type(fine_tuned_model)}")

# Test the fine-tuned model
print("\nTesting the fine-tuned model...")
test_statements = [
    "I suspect this person will deceive me",  # Cognitive
    "I feel anxious around this person",      # Affective  
    "I avoid working with this person"       # Behavioral
]

# Get embeddings
embeddings = fine_tuned_model.encode(test_statements)
print(f"Generated embeddings shape: {embeddings.shape}")

# Calculate similarity between statements
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(embeddings)
print(f"Similarity matrix:\n{similarity_matrix}")

print("Fine-tuning example completed!")


# EFA_item_selection
print("\n" + "=" * 60)
print("Performing EFA item selection...")

print(f"Using top {validated_items['metadata']['top_n_per_dimension']} items per dimension from validation")

# Use the top validated items directly for EFA analysis
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






