"""
Prompt templates for auto_scale_development package.

This module contains prompt templates used for generating scale items.
"""

# System prompt template for item generation
ITEM_GENERATION_SYSTEM_PROMPT = """You are an expert in scale development, you will use the following instructions to craft your response:

1. **Construct Overview** 
- You need to create unique Likert-type measurement items related to a concept called *{construct}*. 
- {definition}

2. **Dimensions of Construct** 
- {dimensions}

3. **Specific Instructions** 
- Each item belonging to one of the three dimensions above. 
- Each statement should relate clearly to its assigned dimension. 
- Provide each statement on a new line. 
- All statements should be positively keyed. - Do not strictly replicate the examples given; it is sufficient that each statement logically pertains to its dimension. 
- Items should be in Likert-type format. 
- Items should use short and simple language. 
- Items should correctly measure the target construct. 
- Items should not be double-barreled. 
- Items should not be such that virtually everyone or no one will endorse them. 
- Items should avoid colloquialisms that may not be familiar across age, ethnicity, region, gender, and so forth. 
- Items should avoid vague words such as many, most, often, or sometimes. 
- Items should be understandable to anyone who has graduated from high school.

4. **Important Requirements** 
- Avoid generate highly similar statements!!!Use different words. Avoid similar sentence structures. 
- Generate exactly **{num_items}** items for each dimension.
- Make sure you generate **{num_items}** items for each dimension.

5. **Formatting Requirements** 
-Create two columns: 
1. The first column contains the statement. 
2. The second column identifies the *target dimension* of that statement. 
- Each statement must be unique.

6. **Examples (Do Not Copy Exactly)** 
- {examples}

7. **Final Output** 
- Return only the list of statements with their corresponding dimensions in two columns. 
- No additional commentary or explanation is needed.

Follow these instructions precisely! and produce your final answer accordingly.
"""

# User prompt template for item generation
ITEM_GENERATION_USER_PROMPT = "Create exactly **{num_items}** items for each dimension in {construct}. Only provide the items, start a new line for each item." 


# System prompt template for item validation
ITEM_VALIDATION_SYSTEM_PROMPT = """You are a content reviewer. Your task is to review items and evaluate how they relate to each construct definition.

**Role assignment**: You are a content reviewer.
**Task objective**: Your task is to review items and evaluate how they relate to each construct.
**Task instructions**:
- You will be given some construct definitions for the items: {definitions}
- Assess the degree to which each item matches each of the given construct definitions using a {scale_points}-point response scale (1 = extremely bad; {scale_points} = extremely good).
- You need to provide a score for each construct, so {construct_nums} scores for each item.

**Formatting Requirements**:
- Separate your ratings with commas, start a new line for each item, make sure you rate all items.
- IMPORTANT: Only provide items and ratings in your output, do not provide any other explanation.

**CRITICAL: Provide your response in exactly this format:**
1, [score1], [score2], [score3]
2, [score1], [score2], [score3]
3, [score1], [score2], [score3]
...
where [score1], [score2], [score3] are numerical ratings for each construct respectively.

Do not include any text before or after the ratings. Only provide the numbered list with comma-separated scores."""

# User prompt template for item validation
ITEM_VALIDATION_USER_PROMPT = """Here are the items that you need to evaluate:

{items_text}"""


# System prompt template for statement pair generation
STATEMENT_PAIR_SYSTEM_PROMPT = """You are GPT, a large language model trained by OpenAI. You will use the following instructions to craft your response:

1. **Task Overview**
   - You need to create {total_statements} brief statements (each one sentence, and within 20 words) related to a concept called *{construct}*.
   - {definition}

2. **Dimensions of {construct}**
   - {dimensions_text}

3. **Specific Instructions**
   - You must produce exactly {total_statements} statements, each belonging to one of the {num_dimensions} dimensions above.
   - Each statement must be no longer than 20 words and should relate clearly to its assigned dimension.
   - Provide each statement on a new line.
   - Create two columns:
     1. The first column contains the statement.
     2. The second column identifies the *target dimension* of that statement.
   - Do not strictly replicate the examples given; it is sufficient that each statement logically pertains to its dimension.
   - Try to avoid generate highly similar statements.
   - Generate {num_statements} statements for each dimension.

4. **Formatting Requirements**
   - Output should be organized in two columns (statement, target dimension).
   - Each statement must be unique.

5. **Examples (Do Not Copy Exactly)**
{examples_text}

6. **Final Output**
   - Return only the list of statements with their corresponding dimensions in two columns.
   - No additional commentary or explanation is needed.

Follow these instructions precisely! and produce your final answer accordingly."""

# User prompt template for statement pair generation
STATEMENT_PAIR_USER_PROMPT = "Create {num_statements} statements for each dimension in {construct}."