from auto_scale_development import item_generation, export_items_to_excel, export_items_to_json, item_reduction, content_validation

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
    num_items=5,
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
    models=["chatgpt-4o-latest", "gpt-4.1-2025-04-14"], # Using a faster model for testing
    runs_per_model=2,
    top_n_per_dimension=3,
    api_key=None  # Will use environment variable
)

print("Validated Items:")
print(validated_items)