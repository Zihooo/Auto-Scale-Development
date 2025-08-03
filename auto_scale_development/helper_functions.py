"""
Helper functions for auto_scale_development package.

This module contains utility and helper functions used by the core functions.
"""

from typing import List, Tuple, Dict, Any, Optional
import re
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .config import call_llm_api
from .prompts import ITEM_GENERATION_SYSTEM_PROMPT, ITEM_GENERATION_USER_PROMPT


def _parse_items_output(items_text: str) -> List[Tuple[str, str]]:
    """
    Parse the output from item_generation function into a structured format.
    
    Args:
        items_text (str): The raw text output from item_generation function
    
    Returns:
        List[Tuple[str, str]]: List of tuples containing (item_statement, dimension)
    
    Example:
        >>> items_text = "I trust this person completely.\tCognitive"
        >>> parsed = _parse_items_output(items_text)
        >>> print(parsed)  # [("I trust this person completely.", "Cognitive")]
    """
    parsed_items = []
    
    # Split by lines and process each line
    lines = items_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip markdown table header and separator lines
        if line.startswith('|') and ('---' in line or 'Statement' in line or 'Dimension' in line):
            continue
            
        # Handle markdown table format: | Statement | Dimension |
        if line.startswith('|') and line.endswith('|'):
            # Remove leading and trailing pipes
            line = line[1:-1].strip()
            # Split by pipe and clean up
            parts = [part.strip() for part in line.split('|')]
            if len(parts) >= 2:
                item_statement = parts[0].strip()
                dimension = parts[1].strip()
                if item_statement and dimension and item_statement.lower() != 'statement':
                    parsed_items.append((item_statement, dimension))
                continue
            
        # Try different separators that might be used
        separators = ['\t', '|', ' - ', ' : ', '    ']  # tab, pipe, dash, colon, multiple spaces
        
        item_statement = ""
        dimension = ""
        
        for sep in separators:
            if sep in line:
                parts = line.split(sep, 1)  # Split only on first occurrence
                if len(parts) == 2:
                    item_statement = parts[0].strip()
                    dimension = parts[1].strip()
                    break
        
        # If no separator found, try to extract dimension from the end
        if not item_statement and not dimension:
            # Look for dimension names at the end of the line
            dimension_patterns = [
                r'\s+(Cognitive|Behavioral|Affective)$',
                r'\s+(Cognitive|Behavioral|Affective)\s*$'
            ]
            
            for pattern in dimension_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    dimension = match.group(1)
                    item_statement = line[:match.start()].strip()
                    break
        
        # If still no dimension found, skip this line
        if not item_statement or not dimension:
            continue
            
        parsed_items.append((item_statement, dimension))
    
    return parsed_items


def _drop_highly_similar(embeddings, items, thresh=0.8, verbose=False):
    """
    Iteratively drop the first vector in any pair whose cosine similarity
    exceeds `thresh`. Stops when all remaining pairs are ≤ `thresh`.

    Parameters
    ----------
    embeddings : (n, dim) array-like
        The embedding vectors.
    items : list
        The original items list.
    thresh : float, optional (default = 0.8)
        Similarity cut-off.
    verbose : bool, optional
        If True, prints which indices are removed at each iteration.

    Returns
    -------
    np.ndarray
        Filtered embeddings (order preserved for kept items).
    list[int]
        Original indices of the kept items.
    """
    emb = np.asarray(embeddings)
    keep = np.arange(len(emb))           # original indices still alive

    while True:
        sims = cosine_similarity(emb[keep])   # (m × m) matrix
        sims[np.triu_indices_from(sims)] = 0  # zero out self + upper tri -> look once

        # find a pair (i,j) with similarity > thresh
        i, j = np.where(sims > thresh)
        if len(i) == 0:                      # nothing left to drop
            break

        first_to_drop = keep[np.min(i)]      # the lower index (first item)
        if verbose:
            print(f"Dropping item: '{items[first_to_drop][:50]}...' "
                  f"(similarity {sims[i[0], j[0]]:.3f} with item: '{items[keep[j[0]]][:50]}...')")
        keep = keep[keep != first_to_drop]   # remove it and loop again

    return embeddings[keep], keep.tolist()


def _load_items_from_excel(file_path: str) -> Dict[str, Any]:
    """
    Load items from Excel file.
    
    Args:
        file_path (str): Path to Excel file
    
    Returns:
        Dict[str, Any]: Dictionary with items data
    
    Raises:
        ValueError: If required columns are missing
        Exception: If file cannot be read
    
    Example:
        >>> items_dict = _load_items_from_excel("items.xlsx")
        >>> print(items_dict)
    """
    try:
        # Read Excel file
        df = pd.read_excel(file_path)
        
        # Check required columns
        required_columns = ['Item_Statement', 'Dimension']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Excel file must contain columns: {required_columns}")
        
        # Convert to items format
        items = []
        for index, row in df.iterrows():
            item = {
                "item_number": row.get('Item_Number', index + 1),
                "statement": row['Item_Statement'],
                "dimension": row['Dimension']
            }
            items.append(item)
        
        return {"items": items}
        
    except Exception as e:
        raise Exception(f"Failed to load Excel file: {str(e)}")


def _load_items_from_json(file_path: str) -> Dict[str, Any]:
    """
    Load items from JSON file.
    
    Args:
        file_path (str): Path to JSON file
    
    Returns:
        Dict[str, Any]: Dictionary with items data
    
    Raises:
        ValueError: If file format is invalid
        Exception: If file cannot be read
    
    Example:
        >>> items_dict = _load_items_from_json("items.json")
        >>> print(items_dict)
    """
    try:
        # Read JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # If data already has 'items' key, return as is
        if "items" in data:
            return data
        
        # If data is a list, assume it's items list
        if isinstance(data, list):
            return {"items": data}
        
        # If data is a dictionary but no 'items' key, try to find items
        if isinstance(data, dict):
            # Look for common keys that might contain items
            for key in ['items', 'data', 'scale_items', 'statements']:
                if key in data and isinstance(data[key], list):
                    return {"items": data[key]}
        
        raise ValueError("JSON file must contain 'items' key or be a list of items")
        
    except Exception as e:
        raise Exception(f"Failed to load JSON file: {str(e)}")


def _parse_validation_ratings(response: str, num_items: int, construct_nums: int, scale_points: int) -> List[List[float]]:
    """
    Parse validation ratings from LLM response.
    
    Args:
        response (str): Raw response from LLM
        num_items (int): Number of items expected
        construct_nums (int): Number of constructs/dimensions
        scale_points (int): Maximum scale points
    
    Returns:
        List[List[float]]: Parsed ratings for each item
    """
    try:
        lines = response.strip().split('\n')
        ratings = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip lines that don't start with a number
            if not line[0].isdigit():
                continue
                
            # Try different parsing patterns
            # Pattern 1: "1, 6.5, 2.1, 3.2"
            if ',' in line:
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= construct_nums + 1:  # +1 for item number
                    try:
                        # Skip item number, get scores
                        scores = [float(parts[i]) for i in range(1, construct_nums + 1)]
                        # Validate scores are within reasonable range (1 to scale_points*1.5)
                        if all(1 <= score <= scale_points * 1.5 for score in scores):
                            ratings.append(scores)
                    except ValueError:
                        continue
            
            # Pattern 2: "1. 6.5 2.1 3.2" (period separator)
            elif '.' in line and len(line.split('.')) >= 2:
                parts = line.split('.', 1)[1].strip().split()
                if len(parts) >= construct_nums:
                    try:
                        scores = [float(parts[i]) for i in range(construct_nums)]
                        if all(1 <= score <= scale_points * 1.5 for score in scores):
                            ratings.append(scores)
                    except ValueError:
                        continue
            
            # Pattern 3: "1 6.5 2.1 3.2" (space separator)
            else:
                parts = line.split()
                if len(parts) >= construct_nums + 1:  # +1 for item number
                    try:
                        # Skip item number, get scores
                        scores = [float(parts[i]) for i in range(1, construct_nums + 1)]
                        if all(1 <= score <= scale_points * 1.5 for score in scores):
                            ratings.append(scores)
                    except ValueError:
                        continue
        
        # Ensure we have ratings for all items
        if len(ratings) == num_items:
            return ratings
        else:
            print(f"Warning: Expected {num_items} ratings, got {len(ratings)}")
            return []
            
    except Exception as e:
        print(f"Error parsing ratings: {str(e)}")
        return []


def _calculate_average_ratings(all_ratings: List[List[List[float]]], num_items: int, construct_nums: int) -> List[List[float]]:
    """
    Calculate average ratings across all runs.
    
    Args:
        all_ratings (List[List[List[float]]]): All ratings from multiple runs
        num_items (int): Number of items
        construct_nums (int): Number of constructs/dimensions
    
    Returns:
        List[List[float]]: Average ratings for each item
    """
    # Initialize sum matrix
    sum_ratings = [[0.0 for _ in range(construct_nums)] for _ in range(num_items)]
    
    # Sum all ratings
    for ratings in all_ratings:
        for i in range(num_items):
            for j in range(construct_nums):
                sum_ratings[i][j] += ratings[i][j]
    
    # Calculate averages
    num_runs = len(all_ratings)
    avg_ratings = []
    for i in range(num_items):
        avg_ratings.append([sum_ratings[i][j] / num_runs for j in range(construct_nums)])
    
    return avg_ratings


def _calculate_scores(items: List[Dict[str, Any]], avg_ratings: List[List[float]], dimensions: List[str], scale_points: int) -> List[Dict[str, Any]]:
    """
    Calculate correspondence and distinctiveness scores for each item.
    
    Args:
        items (List[Dict[str, Any]]): List of items
        avg_ratings (List[List[float]]): Average ratings for each item
        dimensions (List[str]): List of dimension names
        scale_points (int): Maximum scale points
    
    Returns:
        List[Dict[str, Any]]: Items with calculated scores
    """
    validated_items = []
    
    for i, item in enumerate(items):
        item_dimension = item.get("dimension")
        ratings = avg_ratings[i]
        
        # Find the index of the item's dimension
        try:
            correspondence_idx = dimensions.index(item_dimension)
            correspondence_rating = ratings[correspondence_idx]
            
            # Calculate orbiting correspondence ratings (other dimensions)
            orbiting_ratings = [ratings[j] for j in range(len(ratings)) if j != correspondence_idx]
            avg_orbiting_rating = sum(orbiting_ratings) / len(orbiting_ratings) if orbiting_ratings else 0
            
            # Calculate c_value and d_value
            c_value = correspondence_rating / scale_points
            d_value = (correspondence_rating - avg_orbiting_rating) / (scale_points - 1)
            
            # Create enhanced item with scores
            enhanced_item = item.copy()
            enhanced_item.update({
                "correspondence_rating": correspondence_rating,
                "orbiting_correspondence_ratings": orbiting_ratings,
                "avg_orbiting_rating": avg_orbiting_rating,
                "c_value": c_value,
                "d_value": d_value,
                "all_ratings": dict(zip(dimensions, ratings))
            })
            
            validated_items.append(enhanced_item)
            
        except ValueError:
            # If dimension not found, skip this item
            print(f"Warning: Dimension '{item_dimension}' not found in definitions for item {i+1}")
            continue
    
    return validated_items


def _select_top_items(validated_items: List[Dict[str, Any]], dimensions: List[str], top_n_per_dimension: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    Select top items per dimension based on c_value and d_value averages.
    
    Args:
        validated_items (List[Dict[str, Any]]): List of validated items with scores
        dimensions (List[str]): List of dimension names
        top_n_per_dimension (int): Number of top items to select per dimension
    
    Returns:
        Dict[str, List[Dict[str, Any]]]: Top items grouped by dimension
    """
    top_items = {dim: [] for dim in dimensions}
    
    # Group items by dimension
    items_by_dimension = {dim: [] for dim in dimensions}
    for item in validated_items:
        dim = item.get("dimension")
        if dim in items_by_dimension:
            items_by_dimension[dim].append(item)
    
    # Select top items for each dimension
    for dim in dimensions:
        items = items_by_dimension[dim]
        if items:
            # Sort by average of c_value and d_value
            sorted_items = sorted(items, key=lambda x: (x["c_value"] + x["d_value"]) / 2, reverse=True)
            top_items[dim] = sorted_items[:top_n_per_dimension]
    
    return top_items


def _adjust_items_per_dimension(
    parsed_items: List[Tuple[str, str]], 
    dimensions: Dict[str, str], 
    target_num_items: int,
    examples: Optional[Dict[str, list]] = None,
    model_name: str = "gpt-4.1-2025-04-14",
    temperature: float = 1.0,
    top_p: float = 0.8,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    together_api_key: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Adjust the number of items per dimension to match the target number.
    
    Args:
        parsed_items (List[Tuple[str, str]]): List of (statement, dimension) tuples
        dimensions (Dict[str, str]): Dictionary mapping dimension names to descriptions
        target_num_items (int): Target number of items per dimension
        examples (Optional[Dict[str, list]]): Examples for each dimension
        model_name (str): LLM model name
        temperature (float): Temperature for generation
        top_p (float): Top-p for generation
        openai_api_key (Optional[str]): OpenAI API key
        anthropic_api_key (Optional[str]): Anthropic API key
        google_api_key (Optional[str]): Google API key
        together_api_key (Optional[str]): Together API key
    
    Returns:
        List[Tuple[str, str]]: Adjusted list of items
    """
    # Count items per dimension
    items_by_dimension = {}
    for statement, dimension in parsed_items:
        if dimension not in items_by_dimension:
            items_by_dimension[dimension] = []
        items_by_dimension[dimension].append(statement)
    
    adjusted_items = []
    
    for dim_name, dim_desc in dimensions.items():
        current_items = items_by_dimension.get(dim_name, [])
        current_count = len(current_items)
        
        if current_count == target_num_items:
            # Perfect match, keep all items
            for item in current_items:
                adjusted_items.append((item, dim_name))
        
        elif current_count > target_num_items:
            # Too many items, keep only the first target_num_items
            for i in range(target_num_items):
                adjusted_items.append((current_items[i], dim_name))
            # print(f"Dimension '{dim_name}': Removed {current_count - target_num_items} extra items")
        
        elif current_count < target_num_items:
            # Too few items, generate more
            needed_items = target_num_items - current_count
            # print(f"Dimension '{dim_name}': Generating {needed_items} additional items")
            
            # Add existing items
            for item in current_items:
                adjusted_items.append((item, dim_name))
            
            # Generate additional items for this dimension
            additional_items = _generate_additional_items(
                dimension_name=dim_name,
                dimension_desc=dim_desc, 
                num_items=needed_items,
                examples=examples,
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                openai_api_key=openai_api_key,
                anthropic_api_key=anthropic_api_key,
                google_api_key=google_api_key,
                together_api_key=together_api_key
            )
            
            for item in additional_items:
                adjusted_items.append((item, dim_name))
    
    return adjusted_items


def _generate_additional_items(
    dimension_name: str,
    dimension_desc: str,
    num_items: int,
    examples: Optional[Dict[str, list]] = None,
    model_name: str = "gpt-4.1-2025-04-14",
    temperature: float = 1.0,
    top_p: float = 0.8,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    together_api_key: Optional[str] = None
) -> List[str]:
    """
    Generate additional items for a specific dimension.
    
    Args:
        dimension_name (str): Name of the dimension
        dimension_desc (str): Description of the dimension
        num_items (int): Number of additional items to generate
        examples (Optional[Dict[str, list]]): Examples for the dimension
        model_name (str): LLM model name
        temperature (float): Temperature for generation
        top_p (float): Top-p for generation
        openai_api_key (Optional[str]): OpenAI API key
        anthropic_api_key (Optional[str]): Anthropic API key
        google_api_key (Optional[str]): Google API key
        together_api_key (Optional[str]): Together API key
    
    Returns:
        List[str]: List of generated items
    """
    # Create a focused prompt for generating items for this specific dimension
    dimension_examples = examples.get(dimension_name, []) if examples else []
    
    # Format examples string
    if dimension_examples:
        examples_str = ", ".join([f'"{example}"' for example in dimension_examples])
    else:
        examples_str = "No specific examples provided."
    
    # Create focused system prompt
    focused_system_prompt = f"""You are an expert in scale development. Generate exactly {num_items} unique Likert-type measurement items for the dimension "{dimension_name}".

**Dimension Description**: {dimension_desc}

**Examples (Do Not Copy Exactly)**: {examples_str}

**Requirements**:
- Generate exactly {num_items} items for this dimension only
- Each statement should relate clearly to the {dimension_name} dimension
- All statements should be positively keyed
- Items should be in Likert-type format
- Items should use short and simple language
- Items should not be double-barreled
- Items should avoid vague words such as many, most, often, or sometimes
- Items should be understandable to anyone who has graduated from high school
- Avoid generating highly similar statements - use different words and sentence structures

**Format**: Return only the list of statements, one per line, with no additional text or formatting."""

    focused_user_prompt = f"Generate exactly {num_items} items for the {dimension_name} dimension. Only provide the items, one per line."

    try:
        response = call_llm_api(
            model_name=model_name,
            user_prompt=focused_user_prompt,
            system_prompt=focused_system_prompt,
            temperature=temperature,
            top_p=top_p,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            google_api_key=google_api_key,
            together_api_key=together_api_key
        )
        
        # Parse the response to extract just the statements
        lines = response.strip().split('\n')
        items = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('|') and not line.startswith('-') and not line.startswith('*'):
                # Remove any numbering or bullet points
                line = re.sub(r'^\d+\.\s*', '', line)  # Remove "1. " format
                line = re.sub(r'^-\s*', '', line)      # Remove "- " format
                line = re.sub(r'^\*\s*', '', line)     # Remove "* " format
                
                if line:
                    items.append(line)
        
        # Ensure we don't return more items than requested
        return items[:num_items]
        
    except Exception as e:
        print(f"Warning: Failed to generate additional items for dimension '{dimension_name}': {str(e)}")
        return [] 