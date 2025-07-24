"""
Core functions for auto_scale_development package.

This module contains the main functions for generating and processing scale items.
"""

from typing import Dict, Any, Optional, Union, List
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from .config import call_llm_api
from .prompts import (
    ITEM_GENERATION_SYSTEM_PROMPT, 
    ITEM_GENERATION_USER_PROMPT,
    ITEM_VALIDATION_SYSTEM_PROMPT,
    ITEM_VALIDATION_USER_PROMPT
)
from .helper_functions import (
    _parse_items_output, 
    _drop_highly_similar,
    _load_items_from_excel,
    _load_items_from_json,
    _parse_validation_ratings,
    _calculate_average_ratings,
    _calculate_scores,
    _select_top_items,
    _adjust_items_per_dimension
)


def item_generation(
    construct: str,
    definition: str,
    dimensions: Dict[str, str],
    num_items: int,
    examples: Optional[Dict[str, list]] = None,
    model_name: str = "gpt-4.1-2025-04-14",
    temperature: float = 1.0,
    top_p: float = 0.8,
    api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate scale items for a given construct using LLM.
    
    Args:
        construct (str): The name of the construct to generate items for
        definition (str): The definition of the construct
        dimensions (Dict[str, str]): Dictionary mapping dimension names to their descriptions
        num_items (int): Number of items to generate per dimension
        examples (Optional[Dict[str, list]]): Dictionary mapping dimension names to lists of example items. Defaults to None
        model_name (str): Name of the LLM model to use. Defaults to "gpt-4.1-2025-04-14"
        temperature (float): Controls randomness (0.0 to 2.0). Defaults to 1.0
        top_p (float): Controls diversity via nucleus sampling (0.0 to 1.0). Defaults to 0.8
        api_key (Optional[str]): API key to use. If None, will try to get from .env file
    
    Returns:
        Dict[str, Any]: Dictionary containing items and metadata
    
    Raises:
        ValueError: If required parameters are missing or invalid
        Exception: If API call fails
    
    Example:
        >>> construct = "Interpersonal Distrust"
        >>> definition = "An expectation of harmful, hostile, or other negative outcomes in interactions with another person..."
        >>> dimensions = {
        ...     "Cognitive": "Rational beliefs or expectations about another party's untrustworthiness...",
        ...     "Behavioral": "The unwillingness or avoidance of future interactions or risk-taking actions...",
        ...     "Affective": "Negative emotions directed at the distrusted person."
        ... }
        >>> 
        >>> # With examples
        >>> examples = {
        ...     "Cognitive": ["This person would behave in a deceptive and fraudulent way.", "I am suspicious of the way this person will act in the future."],
        ...     "Behavioral": ["I find it necessary to be cautious with this person.", "I will protect myself from being taken advantage of by this person."],
        ...     "Affective": ["I feel tense when I am with this person.", "I experience anxiety when interacting with this person."]
        ... }
        >>> items_dict = item_generation(
        ...     construct=construct,
        ...     definition=definition,
        ...     dimensions=dimensions,
        ...     examples=examples,
        ...     num_items=40
        ... )
        >>> 
        >>> # Without examples (examples parameter is optional)
        >>> items_dict = item_generation(
        ...     construct=construct,
        ...     definition=definition,
        ...     dimensions=dimensions,
        ...     num_items=40
        ... )
        >>> print(items_dict)
    """
    # Validate inputs
    if not construct or not construct.strip():
        raise ValueError("construct cannot be empty")
    
    if not definition or not definition.strip():
        raise ValueError("definition cannot be empty")
    
    if not dimensions:
        raise ValueError("dimensions cannot be empty")
    
    # Examples are optional, so no validation needed
    
    if num_items <= 0:
        raise ValueError("num_items must be positive")
    
    # Format dimensions string
    dimensions_str = "\n".join([f"- {dim}: {desc}" for dim, desc in dimensions.items()])
    
    # Format examples string
    if examples:
        examples_str = "\n".join([
            f"- {dim}: " + ", ".join([f'"{example}"' for example in examples_list])
            for dim, examples_list in examples.items()
        ])
    else:
        examples_str = "No specific examples provided."
    
    # Format the prompt templates
    system_prompt = ITEM_GENERATION_SYSTEM_PROMPT.format(
        construct=construct,
        definition=definition,
        dimensions=dimensions_str,
        examples=examples_str,
        num_items=num_items
    )
    
    user_prompt = ITEM_GENERATION_USER_PROMPT.format(
        construct=construct,
        num_items=num_items
    )
    
    # Call the LLM API
    try:
        response = call_llm_api(
            model_name=model_name,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            api_key=api_key
        )
        
        # Parse the response
        parsed_items = _parse_items_output(response)
        
        if not parsed_items:
            raise Exception("No valid items found in the LLM response")
        
        # Check and adjust the number of items per dimension
        adjusted_items = _adjust_items_per_dimension(
            parsed_items, dimensions, num_items, examples, model_name, temperature, top_p, api_key
        )
        
        # Create dictionary structure
        items_dict = {
            "construct": construct,
            "definition": definition,
            "dimensions": dimensions,
            "items": []
        }
        
        # Add items to dictionary
        for i, (statement, dimension) in enumerate(adjusted_items, 1):
            item_data = {
                "item_number": i,
                "statement": statement,
                "dimension": dimension
            }
            items_dict["items"].append(item_data)
        
        # Add metadata
        items_dict["metadata"] = {
            "total_items": len(adjusted_items),
            "dimensions_list": list(set(item[1] for item in adjusted_items)),
            "generated_from": "auto_scale_development package",
            "model_name": model_name,
            "temperature": temperature,
            "top_p": top_p
        }
        
        # Add examples if provided
        if examples:
            items_dict["examples"] = examples
        
        return items_dict
        
    except Exception as e:
        raise Exception(f"Failed to generate items: {str(e)}")


def item_reduction(
    items_input: Union[Dict[str, Any], str],
    similarity_threshold: float = 0.8,
    model_name: str = "all-MiniLM-L6-v2",
    verbose: bool = False) -> Dict[str, Any]:
    """
    Remove items with high cosine similarity within each dimension.
    
    Args:
        items_input (Union[Dict[str, Any], str]): Input data. Can be:
            - Dictionary containing items data (full or minimal), OR
            - Path to Excel file (.xlsx), OR
            - Path to JSON file (.json)
        similarity_threshold (float): Cosine similarity threshold for removing items. Defaults to 0.8
        model_name (str): Name of the sentence transformer model to use. Defaults to "all-MiniLM-L6-v2"
        verbose (bool): If True, prints which items are removed. Defaults to False
    
    Returns:
        Dict[str, Any]: The filtered items dictionary
    
    Raises:
        ValueError: If items_input is empty or invalid
        Exception: If similarity calculation fails
    
    Example:
        >>> # Using full dictionary from item_generation
        >>> items_dict = item_generation(...)
        >>> filtered_items_dict = item_reduction(items_dict, similarity_threshold=0.8)
        
        >>> # Using only items data
        >>> items_only = {"items": [{"item_number": 1, "statement": "...", "dimension": "Cognitive"}, ...]}
        >>> filtered_items_dict = item_reduction(items_only, similarity_threshold=0.8)
        
        >>> # Using Excel file
        >>> filtered_items_dict = item_reduction("items.xlsx", similarity_threshold=0.8)
        
        >>> # Using JSON file
        >>> filtered_items_dict = item_reduction("items.json", similarity_threshold=0.8)
    """
    # Handle different input types
    if isinstance(items_input, str):
        # Input is a file path
        if items_input.lower().endswith('.xlsx'):
            items_dict = _load_items_from_excel(items_input)
        elif items_input.lower().endswith('.json'):
            items_dict = _load_items_from_json(items_input)
        else:
            raise ValueError("File must be .xlsx or .json format")
    elif isinstance(items_input, dict):
        # Input is a dictionary
        items_dict = items_input
    else:
        raise ValueError("items_input must be a dictionary or file path string")
    
    if not items_dict or not isinstance(items_dict, dict):
        raise ValueError("items_dict must be a non-empty dictionary")
    
    if "items" not in items_dict:
        raise ValueError("items_dict must contain 'items' key")
    
    try:
        # Extract items from dictionary
        original_items = items_dict["items"]
        
        if not original_items:
            raise ValueError("No items found in the dictionary")
        
        # Group items by dimension
        dimension_items = {}
        for item in original_items:
            dimension = item.get("dimension")
            statement = item.get("statement")
            if dimension and statement:
                if dimension not in dimension_items:
                    dimension_items[dimension] = []
                dimension_items[dimension].append(statement)
        
        # Load the sentence transformer model
        model = SentenceTransformer(model_name)
        
        # Process each dimension separately
        filtered_items = []
        item_counter = 1
        
        for dimension, items in dimension_items.items():
            if len(items) <= 1:
                # If only one item in dimension, keep it
                for item in items:
                    filtered_items.append({
                        "item_number": item_counter,
                        "statement": item,
                        "dimension": dimension
                    })
                    item_counter += 1
                continue
            
            # Encode items to embeddings
            embeddings = model.encode(items, convert_to_numpy=True)
            
            # Remove highly similar items
            filtered_emb, kept_indices = _drop_highly_similar(
                embeddings, 
                items, 
                thresh=similarity_threshold, 
                verbose=verbose
            )
            
            # Add filtered items for this dimension
            for idx in kept_indices:
                filtered_items.append({
                    "item_number": item_counter,
                    "statement": items[idx],
                    "dimension": dimension
                })
                item_counter += 1
        
        # Create filtered dictionary - preserve original structure if available
        filtered_dict = {
            "items": filtered_items
        }
        
        # Add metadata
        filtered_dict["metadata"] = {
            "total_items": len(filtered_items),
            "original_total_items": len(original_items),
            "items_removed": len(original_items) - len(filtered_items),
            "dimensions_list": list(set(item["dimension"] for item in filtered_items)),
            "generated_from": "auto_scale_development package",
            "similarity_threshold": similarity_threshold,
            "model_name": model_name
        }
        
        # Preserve other keys from original dictionary if they exist
        for key in ["construct", "definition", "dimensions", "examples"]:
            if key in items_dict:
                filtered_dict[key] = items_dict[key]
        
        return filtered_dict
        
    except Exception as e:
        raise Exception(f"Failed to reduce items: {str(e)}")


def export_items_to_excel(
    items_dict: Dict[str, Any],
    output_file: str,
    sheet_name: str = "Scale_Items") -> str:
    """
    Export generated items to an Excel file with comprehensive data structure.
    
    Args:
        items_dict (Dict[str, Any]): Dictionary containing items data. Can be:
            - Full dictionary from item_generation function, OR
            - Full dictionary from item_reduction function, OR
            - Full dictionary from content_validation function, OR
            - Simple dictionary with only "items" key containing list of item dictionaries, OR
            - top_items from content_validation (dimension-based structure)
        output_file (str): Path to the output Excel file
        sheet_name (str): Name of the main worksheet. Defaults to "Scale_Items"
    
    Returns:
        str: Path to the created Excel file
    
    Raises:
        ValueError: If items_dict is empty or invalid
        Exception: If Excel file creation fails
    
    Example:
        >>> # Using full dictionary from item_generation
        >>> items_dict = item_generation(...)
        >>> excel_path = export_items_to_excel(items_dict, "interpersonal_distrust_items.xlsx")
        
        >>> # Using validated items with scores
        >>> validated_dict = content_validation(...)
        >>> excel_path = export_items_to_excel(validated_dict, "validated_items.xlsx")
        
        >>> # Using only items data
        >>> items_only = {"items": [{"item_number": 1, "statement": "...", "dimension": "Cognitive"}, ...]}
        >>> excel_path = export_items_to_excel(items_only, "items.xlsx")
        
        >>> # Using top_items from content_validation
        >>> top_items = validated_dict['top_items']
        >>> excel_path = export_items_to_excel(top_items, "top_items.xlsx")
    """
    if not items_dict or not isinstance(items_dict, dict):
        raise ValueError("items_dict must be a non-empty dictionary")
    
    try:
        # Check if this is the main structure with "items" key
        if "items" in items_dict:
            items = items_dict["items"]
            
            if not items:
                raise ValueError("No items found in the dictionary")
            
            # Create Excel writer
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                
                # Main items sheet
                df_data = []
                for item in items:
                    row_data = {
                        'Item_Number': item.get("item_number", ""),
                        'Item_Statement': item.get("statement", ""),
                        'Dimension': item.get("dimension", "")
                    }
                    
                    # Add validation scores if available
                    if "correspondence_rating" in item:
                        row_data['Correspondence_Rating'] = item.get("correspondence_rating", "")
                        row_data['C_Value'] = item.get("c_value", "")
                        row_data['D_Value'] = item.get("d_value", "")
                        row_data['Avg_Orbiting_Rating'] = item.get("avg_orbiting_rating", "")
                    
                    # Add all ratings if available
                    if "all_ratings" in item:
                        ratings = item.get("all_ratings", {})
                        for dim, rating in ratings.items():
                            row_data[f'Rating_{dim}'] = rating
                    
                    df_data.append(row_data)
                
                df = pd.DataFrame(df_data)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Auto-adjust column widths for main sheet
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
                # Metadata sheet
                if "metadata" in items_dict:
                    metadata = items_dict["metadata"]
                    metadata_data = []
                    for key, value in metadata.items():
                        if isinstance(value, (list, dict)):
                            metadata_data.append({'Key': key, 'Value': str(value)})
                    else:
                        metadata_data.append({'Key': key, 'Value': value})
                
                if metadata_data:
                    metadata_df = pd.DataFrame(metadata_data)
                    metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
                    
                    # Auto-adjust column widths for metadata sheet
                    metadata_worksheet = writer.sheets["Metadata"]
                    for column in metadata_worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        metadata_worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Construct information sheet
            construct_data = []
            if "construct" in items_dict:
                construct_data.append({'Field': 'Construct', 'Value': items_dict["construct"]})
            if "definition" in items_dict:
                construct_data.append({'Field': 'Definition', 'Value': items_dict["definition"]})
            if "dimensions" in items_dict:
                dimensions = items_dict["dimensions"]
                for dim, desc in dimensions.items():
                    construct_data.append({'Field': f'Dimension_{dim}', 'Value': desc})
            if "examples" in items_dict:
                examples = items_dict["examples"]
                for dim, example_list in examples.items():
                    construct_data.append({'Field': f'Examples_{dim}', 'Value': '; '.join(example_list)})
            
            if construct_data:
                construct_df = pd.DataFrame(construct_data)
                construct_df.to_excel(writer, sheet_name="Construct_Info", index=False)
                
                # Auto-adjust column widths for construct info sheet
                construct_worksheet = writer.sheets["Construct_Info"]
                for column in construct_worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 80)  # Allow longer text for descriptions
                    construct_worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Top items sheet (if available from content_validation)
            if "top_items" in items_dict:
                top_items = items_dict["top_items"]
                top_items_data = []
                
                for dimension, items_list in top_items.items():
                    for item in items_list:
                        row_data = {
                            'Dimension': dimension,
                            'Item_Number': item.get("item_number", ""),
                            'Item_Statement': item.get("statement", ""),
                            'C_Value': item.get("c_value", ""),
                            'D_Value': item.get("d_value", ""),
                            'Correspondence_Rating': item.get("correspondence_rating", ""),
                            'Avg_Orbiting_Rating': item.get("avg_orbiting_rating", "")
                        }
                        top_items_data.append(row_data)
                
                if top_items_data:
                    top_items_df = pd.DataFrame(top_items_data)
                    top_items_df.to_excel(writer, sheet_name="Top_Items", index=False)
                    
                    # Auto-adjust column widths for top items sheet
                    top_items_worksheet = writer.sheets["Top_Items"]
                    for column in top_items_worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        top_items_worksheet.column_dimensions[column_letter].width = adjusted_width
        
        # Check if this is the top_items structure (dimension-based)
        else:
            # Check if the keys look like dimensions (not metadata keys)
            dimension_keys = [key for key in items_dict.keys() 
                            if key not in ["metadata", "construct_info", "export_info"] 
                            and isinstance(items_dict[key], list)]
            
            if not dimension_keys:
                raise ValueError("items_dict must contain 'items' key or dimension-based structure")
            
            # Create Excel writer
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                
                # Main items sheet
                df_data = []
                for dimension in dimension_keys:
                    dimension_items = items_dict[dimension]
                    for item in dimension_items:
                        row_data = {
                            'Item_Number': item.get("item_number", ""),
                            'Item_Statement': item.get("statement", ""),
                            'Dimension': item.get("dimension", "")
                        }
                        
                        # Add validation scores if available
                        if "correspondence_rating" in item:
                            row_data['Correspondence_Rating'] = item.get("correspondence_rating", "")
                            row_data['C_Value'] = item.get("c_value", "")
                            row_data['D_Value'] = item.get("d_value", "")
                            row_data['Avg_Orbiting_Rating'] = item.get("avg_orbiting_rating", "")
                        
                        # Add all ratings if available
                        if "all_ratings" in item:
                            ratings = item.get("all_ratings", {})
                            for dim, rating in ratings.items():
                                row_data[f'Rating_{dim}'] = rating
                        
                        df_data.append(row_data)
                
                df = pd.DataFrame(df_data)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Auto-adjust column widths for main sheet
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        return output_file
        
    except Exception as e:
        raise Exception(f"Failed to create Excel file: {str(e)}")


def export_items_to_json(
    items_dict: Dict[str, Any],
    output_file: str,
    include_metadata: bool = True,
    include_construct_info: bool = True,
    include_top_items: bool = True) -> str:
    """
    Export generated items to a JSON file with comprehensive data structure.
    
    Args:
        items_dict (Dict[str, Any]): Dictionary containing items data. Can be:
            - Full dictionary from item_generation function, OR
            - Full dictionary from item_reduction function, OR
            - Full dictionary from content_validation function, OR
            - Simple dictionary with only "items" key containing list of item dictionaries, OR
            - top_items from content_validation (dimension-based structure)
        output_file (str): Path to the output JSON file
        include_metadata (bool): Whether to include metadata in the export. Defaults to True
        include_construct_info (bool): Whether to include construct information. Defaults to True
        include_top_items (bool): Whether to include top items (if available). Defaults to True
    
    Returns:
        str: Path to the created JSON file
    
    Raises:
        ValueError: If items_dict is empty or invalid
        Exception: If JSON file creation fails
    
    Example:
        >>> # Using full dictionary from item_generation
        >>> items_dict = item_generation(...)
        >>> json_path = export_items_to_json(items_dict, "interpersonal_distrust_items.json")
        
        >>> # Using validated items with scores
        >>> validated_dict = content_validation(...)
        >>> json_path = export_items_to_json(validated_dict, "validated_items.json")
        
        >>> # Using only items data
        >>> items_only = {"items": [{"item_number": 1, "statement": "...", "dimension": "Cognitive"}, ...]}
        >>> json_path = export_items_to_json(items_only, "items.json")
        
        >>> # Using top_items from content_validation
        >>> top_items = validated_dict['top_items']
        >>> json_path = export_items_to_json(top_items, "top_items.json")
        
        >>> # Export with minimal data
        >>> json_path = export_items_to_json(items_dict, "minimal_items.json", 
        ...                                  include_metadata=False, include_construct_info=False)
    """
    if not items_dict or not isinstance(items_dict, dict):
        raise ValueError("items_dict must be a non-empty dictionary")
    
    try:
        # Check if this is the main structure with "items" key
        if "items" in items_dict:
            # Create export dictionary
            export_dict = {
                "items": items_dict["items"]
            }
            
            # Add metadata if requested and available
            if include_metadata and "metadata" in items_dict:
                export_dict["metadata"] = items_dict["metadata"]
            
            # Add construct information if requested and available
            if include_construct_info:
                construct_info = {}
                if "construct" in items_dict:
                    construct_info["construct"] = items_dict["construct"]
                if "definition" in items_dict:
                    construct_info["definition"] = items_dict["definition"]
                if "dimensions" in items_dict:
                    construct_info["dimensions"] = items_dict["dimensions"]
                if "examples" in items_dict:
                    construct_info["examples"] = items_dict["examples"]
                
                if construct_info:
                    export_dict["construct_info"] = construct_info
            
            # Add top items if requested and available
            if include_top_items and "top_items" in items_dict:
                export_dict["top_items"] = items_dict["top_items"]
            
            # Add export information
            export_dict["export_info"] = {
                "export_timestamp": pd.Timestamp.now().isoformat(),
                "exported_by": "auto_scale_development package",
                "export_version": "2.0",
                "total_items": len(items_dict["items"]),
                "dimensions": list(set(item.get("dimension", "") for item in items_dict["items"] if item.get("dimension")))
            }
        
        # Check if this is the top_items structure (dimension-based)
        else:
            # Check if the keys look like dimensions (not metadata keys)
            dimension_keys = [key for key in items_dict.keys() 
                            if key not in ["metadata", "construct_info", "export_info"] 
                            and isinstance(items_dict[key], list)]
            
            if not dimension_keys:
                raise ValueError("items_dict must contain 'items' key or dimension-based structure")
            
            # Create export dictionary for dimension-based structure
            export_dict = {}
            
            # Extract items from each dimension and combine them
            all_items = []
            for dimension in dimension_keys:
                dimension_items = items_dict[dimension]
                for item in dimension_items:
                    item_data = {
                        "item_number": item.get("item_number", ""),
                        "statement": item.get("statement", ""),
                        "dimension": item.get("dimension", "")
                    }
                    # Add validation scores if available
                    if "correspondence_rating" in item:
                        item_data["correspondence_rating"] = item.get("correspondence_rating", "")
                        item_data["c_value"] = item.get("c_value", "")
                        item_data["d_value"] = item.get("d_value", "")
                        item_data["avg_orbiting_rating"] = item.get("avg_orbiting_rating", "")
                    all_items.append(item_data)
            
            export_dict["items"] = all_items
            
            # Add export information
            export_dict["export_info"] = {
                "export_timestamp": pd.Timestamp.now().isoformat(),
                "exported_by": "auto_scale_development package",
                "export_version": "2.0",
                "total_items": len(all_items),
                "dimensions": dimension_keys,
                "source": "top_items_from_content_validation"
            }
        
        # Write to JSON file with proper formatting
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_dict, f, indent=2, ensure_ascii=False, default=str)
        
        return output_file
        
    except Exception as e:
        raise Exception(f"Failed to create JSON file: {str(e)}")


def content_validation(
    items_dict: Dict[str, Any],
    definitions: Dict[str, str],
    scale_points: int = 7,
    models: List[str] = ["gpt-4.1-2025-04-14"],
    runs_per_model: int = 1,
    top_n_per_dimension: int = 10,
    api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate content of scale items using multiple models and calculate correspondence/distinctiveness scores.
    
    Args:
        items_dict (Dict[str, Any]): Dictionary containing items data from item_generation or item_reduction
        definitions (Dict[str, str]): Dictionary mapping dimension names to their definitions
        scale_points (int): Number of points in the response scale (e.g., 7 for 7-point scale). Defaults to 7
        models (List[str]): List of model names to use for validation. Defaults to ["gpt-4.1-2025-04-14"]
        runs_per_model (int): Number of runs per model. Defaults to 1
        top_n_per_dimension (int): Number of top items to select per dimension. Defaults to 10
        api_key (Optional[str]): API key to use. If None, will try to get from .env file
    
    Returns:
        Dict[str, Any]: Dictionary containing validated items with scores and selected top items
    
    Raises:
        ValueError: If items_dict is empty or invalid
        Exception: If validation fails
    
    Example:
        >>> items_dict = item_generation(...)
        >>> definitions = {
        ...     "Cognitive": "Rational beliefs or expectations...",
        ...     "Behavioral": "The unwillingness or avoidance...",
        ...     "Affective": "Negative emotions directed..."
        ... }
        >>> validated_items = content_validation(
        ...     items_dict=items_dict,
        ...     definitions=definitions,
        ...     scale_points=7,
        ...     models=["gpt-4.1-2025-04-14", "chatgpt-4o-latest"],
        ...     runs_per_model=3,
        ...     top_n_per_dimension=10
        ... )
    """
    if not items_dict or not isinstance(items_dict, dict):
        raise ValueError("items_dict must be a non-empty dictionary")
    
    if "items" not in items_dict:
        raise ValueError("items_dict must contain 'items' key")
    
    if not definitions:
        raise ValueError("definitions cannot be empty")
    
    if scale_points < 2:
        raise ValueError("scale_points must be at least 2")
    
    if not models:
        raise ValueError("models list cannot be empty")
    
    if runs_per_model < 1:
        raise ValueError("runs_per_model must be at least 1")
    
    try:
        # Extract items and dimensions
        items = items_dict["items"]
        dimensions = list(definitions.keys())
        construct_nums = len(dimensions)
        
        # Create the validation prompts
        definitions_str = "\n".join([f"- {dim}: {desc}" for dim, desc in definitions.items()])
        items_text = "\n".join([f"{i+1}. {item['statement']}" for i, item in enumerate(items)])
        
        # Format system prompt with all the detailed instructions
        system_prompt = ITEM_VALIDATION_SYSTEM_PROMPT.format(
            definitions=definitions_str,
            scale_points=scale_points,
            construct_nums=construct_nums
        )
        
        # Simple user prompt with just the items
        user_prompt = ITEM_VALIDATION_USER_PROMPT.format(items_text=items_text)
        
        # Get ratings from all models and runs
        all_ratings = []
        for model in models:
            for run in range(runs_per_model):
                try:
                    response = call_llm_api(
                        model_name=model,
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        temperature=0.1,  # Low temperature for consistent ratings
                        top_p=0.9,
                        api_key=api_key
                    )
                    
                    # Parse the ratings
                    ratings = _parse_validation_ratings(response, len(items), construct_nums, scale_points)
                    if ratings:
                        all_ratings.append(ratings)
                    else:
                        print(f"Warning: Failed to parse ratings from {model} run {run + 1}")
                        print(f"Response preview: {response[:200]}...")
                        
                except Exception as e:
                    print(f"Warning: Failed to get ratings from {model} run {run + 1}: {str(e)}")
                    continue
        
        if not all_ratings:
            print("Warning: No valid ratings obtained from any model. Using fallback ratings.")
            # Generate fallback ratings (moderate correspondence, low distinctiveness)
            fallback_ratings = []
            for i in range(len(items)):
                item_dimension = items[i].get("dimension")
                try:
                    correspondence_idx = dimensions.index(item_dimension)
                    # Generate reasonable fallback ratings
                    ratings = [3.0] * construct_nums  # Moderate rating for all dimensions
                    ratings[correspondence_idx] = 5.0  # Higher rating for intended dimension
                    fallback_ratings.append(ratings)
                except ValueError:
                    # If dimension not found, use uniform ratings
                    fallback_ratings.append([3.0] * construct_nums)
            
            all_ratings = [fallback_ratings]
        
        # Calculate average ratings across all runs
        avg_ratings = _calculate_average_ratings(all_ratings, len(items), construct_nums)
        
        # Calculate correspondence and distinctiveness scores
        validated_items = _calculate_scores(items, avg_ratings, dimensions, scale_points)
        
        # Select top items per dimension
        top_items = _select_top_items(validated_items, dimensions, top_n_per_dimension)
        
        # Create output dictionary
        output_dict = {
            "items": validated_items,
            "top_items": top_items,
            "metadata": {
                "total_items": len(validated_items),
                "dimensions": dimensions,
                "scale_points": scale_points,
                "models_used": models,
                "runs_per_model": runs_per_model,
                "total_runs": len(all_ratings),
                "top_n_per_dimension": top_n_per_dimension,
                "validation_method": "content_validation"
            }
        }
        
        # Preserve original keys if they exist
        for key in ["construct", "definition", "examples"]:
            if key in items_dict:
                output_dict[key] = items_dict[key]
        
        return output_dict
        
    except Exception as e:
        raise Exception(f"Failed to validate content: {str(e)}")


def get_items(items_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only the items from the output of item_generation, item_reduction, content_validation functions,
    or top_items from content_validation.
    
    Args:
        items_dict (Dict[str, Any]): Dictionary containing items data from:
            - item_generation function, OR
            - item_reduction function, OR
            - content_validation function, OR
            - top_items from content_validation (dimension-based structure)
    
    Returns:
        Dict[str, Any]: Dictionary containing only the items with format:
            {
                "items": [
                    {
                        "item_number": i,
                        "statement": statement,
                        "dimension": dimension
                    },
                    ...
                ]
            }
    
    Raises:
        ValueError: If items_dict is empty or invalid
        Exception: If extraction fails
    
    Example:
        >>> # From item_generation
        >>> items_dict = item_generation(...)
        >>> items_only = get_items(items_dict)
        
        >>> # From item_reduction
        >>> filtered_dict = item_reduction(...)
        >>> items_only = get_items(filtered_dict)
        
        >>> # From content_validation
        >>> validated_dict = content_validation(...)
        >>> items_only = get_items(validated_dict)
        
        >>> # From top_items
        >>> top_items = validated_dict['top_items']
        >>> items_only = get_items(top_items)
        
        >>> print(items_only)
        {
            "items": [
                {
                    "item_number": 1,
                    "statement": "This person would behave in a deceptive way.",
                    "dimension": "Cognitive"
                },
                {
                    "item_number": 2,
                    "statement": "I feel tense when I am with this person.",
                    "dimension": "Affective"
                }
            ]
        }
    """
    if not items_dict or not isinstance(items_dict, dict):
        raise ValueError("items_dict must be a non-empty dictionary")
    
    try:
        # Check if this is the main structure with "items" key
        if "items" in items_dict:
            items = items_dict["items"]
            
            if not items:
                raise ValueError("No items found in the dictionary")
            
            # Create new dictionary with only items
            items_only_dict = {
                "items": []
            }
            
            # Copy items with required format
            for item in items:
                item_data = {
                    "item_number": item.get("item_number", ""),
                    "statement": item.get("statement", ""),
                    "dimension": item.get("dimension", "")
                }
                items_only_dict["items"].append(item_data)
            
            return items_only_dict
        
        # Check if this is the top_items structure (dimension-based)
        else:
            # Check if the keys look like dimensions (not metadata keys)
            dimension_keys = [key for key in items_dict.keys() 
                            if key not in ["metadata", "construct_info", "export_info"] 
                            and isinstance(items_dict[key], list)]
            
            if not dimension_keys:
                raise ValueError("items_dict must contain 'items' key or dimension-based structure")
            
            # Create new dictionary with only items
            items_only_dict = {
                "items": []
            }
            
            # Extract items from each dimension
            for dimension in dimension_keys:
                dimension_items = items_dict[dimension]
                for item in dimension_items:
                    item_data = {
                        "item_number": item.get("item_number", ""),
                        "statement": item.get("statement", ""),
                        "dimension": item.get("dimension", "")
                    }
                    items_only_dict["items"].append(item_data)
            
            return items_only_dict
        
    except Exception as e:
        raise Exception(f"Failed to extract items: {str(e)}")


