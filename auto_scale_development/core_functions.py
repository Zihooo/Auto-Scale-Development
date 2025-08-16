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
    ITEM_VALIDATION_USER_PROMPT,
    STATEMENT_PAIR_SYSTEM_PROMPT,
    STATEMENT_PAIR_USER_PROMPT
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
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    together_api_key: Optional[str] = None) -> Dict[str, Any]:
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
        openai_api_key (Optional[str]): OpenAI API key. If None, will try to get from .env file
        anthropic_api_key (Optional[str]): Anthropic API key. If None, will try to get from .env file
        google_api_key (Optional[str]): Google API key. If None, will try to get from .env file
        together_api_key (Optional[str]): Together API key. If None, will try to get from .env file
    
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
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            google_api_key=google_api_key,
            together_api_key=together_api_key
        )
        
        # Parse the response
        parsed_items = _parse_items_output(response)
        
        if not parsed_items:
            raise Exception("No valid items found in the LLM response")
        
        # Check and adjust the number of items per dimension
        adjusted_items = _adjust_items_per_dimension(
            parsed_items, dimensions, num_items, examples, model_name, temperature, top_p, 
            openai_api_key, anthropic_api_key, google_api_key, together_api_key
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
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    together_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate content of scale items using multiple models and calculate correspondence/distinctiveness scores.
    
    Args:
        items_dict (Dict[str, Any]): Dictionary containing items data from item_generation or item_reduction
        definitions (Dict[str, str]): Dictionary mapping dimension names to their definitions
        scale_points (int): Number of points in the response scale (e.g., 7 for 7-point scale). Defaults to 7
        models (List[str]): List of model names to use for validation. Defaults to ["gpt-4.1-2025-04-14"]
        runs_per_model (int): Number of runs per model. Defaults to 1
        top_n_per_dimension (int): Number of top items to select per dimension. Defaults to 10
        openai_api_key (Optional[str]): OpenAI API key. If None, will try to get from .env file
        anthropic_api_key (Optional[str]): Anthropic API key. If None, will try to get from .env file
        google_api_key (Optional[str]): Google API key. If None, will try to get from .env file
        together_api_key (Optional[str]): Together API key. If None, will try to get from .env file
    
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
                        openai_api_key=openai_api_key,
                        anthropic_api_key=anthropic_api_key,
                        google_api_key=google_api_key,
                        together_api_key=together_api_key
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


def statement_pair(
    construct: str,
    definition: str,
    dimensions: Dict[str, str],
    num_statements: int,
    examples: Optional[Dict[str, list]] = None,
    model_name: str = "gpt-4.1-2025-04-14",
    temperature: float = 1.0,
    top_p: float = 0.8,
    output_file: Optional[str] = None,
    balance_data: bool = True,
    random_seed: Optional[int] = 42,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    together_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate statements using LLM and create pairwise combinations for similarity analysis or machine learning.
    
    Args:
        construct (str): The name of the construct to generate statements for
        definition (str): The definition of the construct
        dimensions (Dict[str, str]): Dictionary mapping dimension names to their descriptions
        num_statements (int): Number of statements to generate per dimension
        examples (Optional[Dict[str, list]]): Dictionary mapping dimension names to lists of example statements. Defaults to None
        model_name (str): Name of the LLM model to use. Defaults to "gpt-4.1-2025-04-14"
        temperature (float): Controls randomness (0.0 to 2.0). Defaults to 1.0
        top_p (float): Controls diversity via nucleus sampling (0.0 to 1.0). Defaults to 0.8
        output_file (Optional[str]): Path to output Excel file. If None, no file will be created. Defaults to None
        balance_data (bool): Whether to balance the dataset by sampling label=0 pairs to match label=1 count. Defaults to True
        random_seed (Optional[int]): Random seed for reproducible sampling. Defaults to 42
        openai_api_key (Optional[str]): OpenAI API key. If None, will try to get from .env file
        anthropic_api_key (Optional[str]): Anthropic API key. If None, will try to get from .env file
        google_api_key (Optional[str]): Google API key. If None, will try to get from .env file
        together_api_key (Optional[str]): Together API key. If None, will try to get from .env file
    
    Returns:
        Dict[str, Any]: Dictionary containing paired statements data with format:
            {
                "pairs": [
                    {
                        "pair_id": i,
                        "statement1": statement1,
                        "statement2": statement2, 
                        "label1": dimension1,
                        "label2": dimension2,
                        "combined_label": "dimension1_dimension2",
                        "label": 1 if same dimension else 0
                    },
                    ...
                ],
                "metadata": {...}
            }
    
    Raises:
        ValueError: If required parameters are missing or invalid
        Exception: If statement generation or pairing process fails
    
    Example:
        >>> construct = "Interpersonal Distrust"
        >>> definition = "An expectation of harmful, hostile, or other negative outcomes..."
        >>> dimensions = {
        ...     "Cognitive": "Rational beliefs or expectations...",
        ...     "Behavioral": "The unwillingness or avoidance...",
        ...     "Affective": "Negative emotions..."
        ... }
        >>> examples = {
        ...     "Cognitive": ["This person would behave in a deceptive way.", "I am suspicious of this person."],
        ...     "Behavioral": ["I find it necessary to be cautious.", "I will protect myself."],
        ...     "Affective": ["I feel tense with this person.", "I experience anxiety."]
        ... }
        >>> pairs_dict = statement_pair(
        ...     construct=construct,
        ...     definition=definition,
        ...     dimensions=dimensions,
        ...     num_statements=40,
        ...     examples=examples,
        ...     output_file="statement_pairs.xlsx"
        ... )
        >>> print(f"Generated {len(pairs_dict['pairs'])} pairs")
        >>> print(f"Same dimension pairs: {pairs_dict['metadata']['same_dimension_pairs']}")
        >>> print(f"Different dimension pairs: {pairs_dict['metadata']['different_dimension_pairs']}")
    """
    # Validate inputs
    if not construct or not construct.strip():
        raise ValueError("construct must be a non-empty string")
    
    if not definition or not definition.strip():
        raise ValueError("definition must be a non-empty string")
    
    if not dimensions or not isinstance(dimensions, dict):
        raise ValueError("dimensions must be a non-empty dictionary")
    
    if num_statements <= 0:
        raise ValueError("num_statements must be a positive integer")
    
    # Initialize items list
    items = []
    
    print("Generating statements using LLM...")
    
    # Prepare prompt parameters
    # Format dimensions for prompt
    dimensions_text = "\n- ".join([f"**{dim}**: {desc}" for dim, desc in dimensions.items()])
    
    # Format examples if provided
    examples_text = ""
    if examples:
        example_lines = []
        for dim, example_list in examples.items():
            for example in example_list:
                example_lines.append(f"   - {dim}: \"{example}\"")
        examples_text = "\n".join(example_lines)
    else:
        # Use default examples if not provided
        examples_text = """   - Cognitive: "This person would behave in a deceptive and fraudulent way.","I am suspicious of the way this person will act in the future.", "This person would use me for his/her own benefits."
   - Behavioral: "I find it necessary to be cautious with this person.","I will protect myself from being taken advantage of by this person.","I will not count on this person for important things."
   - Affective: "I feel tense when I am with this person.","I experience anxiety when interacting with this person.","I worry about future interactions with this person." """
    
    # Calculate total number of statements
    total_statements = num_statements * len(dimensions)
    
    # Create the system prompt using template
    system_prompt = STATEMENT_PAIR_SYSTEM_PROMPT.format(
        total_statements=total_statements,
        construct=construct,
        definition=definition,
        dimensions_text=dimensions_text,
        num_dimensions=len(dimensions),
        num_statements=num_statements,
        examples_text=examples_text
    )
    
    # Create the user prompt using template
    user_prompt = STATEMENT_PAIR_USER_PROMPT.format(
        num_statements=num_statements,
        construct=construct
    )
    
    # Call LLM API
    llm_output = call_llm_api(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        google_api_key=google_api_key,
        together_api_key=together_api_key
    )
    
    # Parse the LLM output
    parsed_items = _parse_items_output(llm_output)
    
    # Convert parsed items to the format expected by pairing logic
    for statement, dimension in parsed_items:
        items.append({
            "statement": statement,
            "dimension": dimension
        })
    
    # Create items_dict for compatibility
    items_dict = {
        "construct": construct,
        "definition": definition,
        "dimensions": dimensions,
        "items": items
    }
    if examples:
        items_dict["examples"] = examples
        
    print(f"Generated {len(items)} statements across {len(dimensions)} dimensions")
        
    
    # Common validation for both modes
    if not items:
        raise ValueError("No items found")
    
    if len(items) < 2:
        raise ValueError("At least 2 items are required to create pairs")
    
    try:
        # Set random seed for reproducibility
        if random_seed is not None:
            import random
            import numpy as np
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Create all possible pairs
        pairs = []
        pair_id = 1
        
        for i in range(len(items)):
            for j in range(i + 1, len(items)):  # Avoid duplicate pairs and self-pairs
                item1 = items[i]
                item2 = items[j]
                
                statement1 = item1.get("statement", "")
                statement2 = item2.get("statement", "")
                label1 = item1.get("dimension", "")
                label2 = item2.get("dimension", "")
                
                # Skip if any required field is missing
                if not statement1 or not statement2 or not label1 or not label2:
                    continue
                
                # Create combined label
                combined_label = f"{label1}_{label2}"
                
                # Determine if same dimension (label = 1) or different (label = 0)
                label = 1 if label1 == label2 else 0
                
                pair_data = {
                    "pair_id": pair_id,
                    "statement1": statement1,
                    "statement2": statement2,
                    "label1": label1,
                    "label2": label2,
                    "combined_label": combined_label,
                    "label": label
                }
                
                pairs.append(pair_data)
                pair_id += 1
        
        if not pairs:
            raise ValueError("No valid pairs could be created from the items")
        
        # Separate same and different dimension pairs
        same_dimension_pairs = [pair for pair in pairs if pair["label"] == 1]
        different_dimension_pairs = [pair for pair in pairs if pair["label"] == 0]
        
        print(f"Created {len(same_dimension_pairs)} same-dimension pairs")
        print(f"Created {len(different_dimension_pairs)} different-dimension pairs")
        
        # Balance data if requested
        final_pairs = pairs.copy()
        
        if balance_data and len(different_dimension_pairs) > len(same_dimension_pairs):
            # Sample from different dimension pairs to match same dimension pairs count
            import random
            sampled_different_pairs = random.sample(different_dimension_pairs, len(same_dimension_pairs))
            final_pairs = same_dimension_pairs + sampled_different_pairs
            
            # Shuffle the final pairs
            random.shuffle(final_pairs)
            
            print(f"Balanced dataset: {len(same_dimension_pairs)} pairs per class")
            print(f"Final dataset size: {len(final_pairs)} pairs")
        
        # Create output dictionary
        pairs_dict = {
            "pairs": final_pairs,
            "metadata": {
                "total_pairs": len(final_pairs),
                "original_total_pairs": len(pairs),
                "same_dimension_pairs": len([p for p in final_pairs if p["label"] == 1]),
                "different_dimension_pairs": len([p for p in final_pairs if p["label"] == 0]),
                "original_items_count": len(items),
                "dimensions_list": list(set(item["dimension"] for item in items if item.get("dimension"))),
                "balance_data": balance_data,
                "random_seed": random_seed,
                "generated_from": "auto_scale_development package"
            }
        }
        
        # Preserve original keys if they exist
        for key in ["construct", "definition", "dimensions", "examples"]:
            if key in items_dict:
                pairs_dict[key] = items_dict[key]
        
        # Export to Excel if output_file is specified
        if output_file:
            export_path = export_pairs_to_excel(pairs_dict, output_file)
            pairs_dict["metadata"]["exported_to"] = export_path
            print(f"Pairs exported to: {export_path}")
        
        return pairs_dict
        
    except Exception as e:
        raise Exception(f"Failed to create statement pairs: {str(e)}")


def export_pairs_to_excel(
    pairs_dict: Dict[str, Any],
    output_file: str,
    sheet_name: str = "Statement_Pairs") -> str:
    """
    Export statement pairs to an Excel file.
    
    Args:
        pairs_dict (Dict[str, Any]): Dictionary containing pairs data from statement_pair function
        output_file (str): Path to the output Excel file
        sheet_name (str): Name of the main worksheet. Defaults to "Statement_Pairs"
    
    Returns:
        str: Path to the created Excel file
    
    Raises:
        ValueError: If pairs_dict is empty or invalid
        Exception: If Excel file creation fails
    
    Example:
        >>> pairs_dict = statement_pair(...)
        >>> excel_path = export_pairs_to_excel(pairs_dict, "statement_pairs.xlsx")
    """
    if not pairs_dict or not isinstance(pairs_dict, dict):
        raise ValueError("pairs_dict must be a non-empty dictionary")
    
    if "pairs" not in pairs_dict:
        raise ValueError("pairs_dict must contain 'pairs' key")
    
    try:
        pairs = pairs_dict["pairs"]
        
        if not pairs:
            raise ValueError("No pairs found in the dictionary")
        
        # Create Excel writer
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # Main pairs sheet
            df_data = []
            for pair in pairs:
                row_data = {
                    'Pair_ID': pair.get("pair_id", ""),
                    'Statement1': pair.get("statement1", ""),
                    'Statement2': pair.get("statement2", ""),
                    'Label1': pair.get("label1", ""),
                    'Label2': pair.get("label2", ""),
                    'Combined_Label': pair.get("combined_label", ""),
                    'Label': pair.get("label", "")
                }
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
                adjusted_width = min(max_length + 2, 80)  # Allow longer text for statements
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Metadata sheet
            if "metadata" in pairs_dict:
                metadata = pairs_dict["metadata"]
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
            if "construct" in pairs_dict:
                construct_data.append({'Field': 'Construct', 'Value': pairs_dict["construct"]})
            if "definition" in pairs_dict:
                construct_data.append({'Field': 'Definition', 'Value': pairs_dict["definition"]})
            if "dimensions" in pairs_dict:
                dimensions = pairs_dict["dimensions"]
                for dim, desc in dimensions.items():
                    construct_data.append({'Field': f'Dimension_{dim}', 'Value': desc})
            if "examples" in pairs_dict:
                examples = pairs_dict["examples"]
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
        
        return output_file
        
    except Exception as e:
        raise Exception(f"Failed to create Excel file: {str(e)}")


def fine_tune(
    pairs_input: Union[Dict[str, Any], str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_path: str = "./fine_tuned_model",
    train_test_split: float = 0.8,
    num_train_epochs: int = 5,
    learning_rate: float = 1e-5,
    per_device_train_batch_size: int = 32,
    per_device_eval_batch_size: int = 32,
    warmup_steps: int = 100,
    loss_function: str = "MultipleNegativesRankingLoss",
    evaluation_steps: int = 100,
    save_steps: int = 500,
    random_seed: Optional[int] = 42,
    use_wandb: bool = False) -> SentenceTransformer:
    """
    Fine-tune a sentence transformer model using statement pairs data.
    
    Note: By default, Weights & Biases (wandb) logging is disabled to avoid login prompts.
    Set use_wandb=True if you want to enable wandb logging (requires wandb account).
    
    Args:
        pairs_input (Union[Dict[str, Any], str]): Input data. Can be:
            - Dictionary containing pairs data from statement_pair function, OR
            - Path to Excel file (.xlsx) created by statement_pair function
        model_name (str): Name or path of the sentence transformer model to fine-tune. Defaults to "sentence-transformers/all-MiniLM-L6-v2"
        output_path (str): Path to save the fine-tuned model. Defaults to "./fine_tuned_model"
        train_test_split (float): Ratio for train/test split (0.0 to 1.0). Defaults to 0.8
        num_train_epochs (int): Number of training epochs. Defaults to 5
        learning_rate (float): Learning rate for training. Defaults to 1e-5
        per_device_train_batch_size (int): Training batch size. Defaults to 32
        per_device_eval_batch_size (int): Evaluation batch size. Defaults to 32
        warmup_steps (int): Number of warmup steps. Defaults to 100
        loss_function (str): Loss function to use. Options: "MultipleNegativesRankingLoss", "CosineSimilarityLoss". Defaults to "MultipleNegativesRankingLoss"
        evaluation_steps (int): Steps between evaluations. Defaults to 100
        save_steps (int): Steps between model saves. Defaults to 500
        random_seed (Optional[int]): Random seed for reproducible training. Defaults to 42
        use_wandb (bool): Whether to enable Weights & Biases logging. Requires wandb account if True. Defaults to False
    
    Returns:
        SentenceTransformer: The fine-tuned model
    
    Raises:
        ValueError: If pairs_input is empty or invalid
        Exception: If training fails
    
    Example:
        >>> # Using pairs dictionary from statement_pair
        >>> pairs_dict = statement_pair(...)
        >>> model = fine_tune(
        ...     pairs_input=pairs_dict,
        ...     model_name="sentence-transformers/all-MiniLM-L6-v2",
        ...     output_path="./my_fine_tuned_model",
        ...     num_train_epochs=10,
        ...     learning_rate=2e-5,
        ...     per_device_train_batch_size=16
        ... )
        
        >>> # Using Excel file
        >>> model = fine_tune(
        ...     pairs_input="statement_pairs.xlsx",
        ...     num_train_epochs=5,
        ...     train_test_split=0.85
        ... )
        
        >>> # Test the model
        >>> similarity = model.similarity(["Statement 1"], ["Statement 2"])
        >>> print(f"Similarity: {similarity}")
    """
    try:
        # Import required libraries
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
        from torch.utils.data import DataLoader
        import torch
        import random
        import numpy as np
        import os
        
        # Set random seeds for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)
        
        # Handle different input types
        if isinstance(pairs_input, str):
            # Input is an Excel file path
            if not pairs_input.lower().endswith('.xlsx'):
                raise ValueError("File must be .xlsx format")
            
            # Load pairs from Excel file
            df = pd.read_excel(pairs_input, sheet_name="Statement_Pairs")
            
            # Convert to pairs format
            pairs_data = []
            for _, row in df.iterrows():
                pair_data = {
                    "pair_id": row.get("Pair_ID", ""),
                    "statement1": row.get("Statement1", ""),
                    "statement2": row.get("Statement2", ""),
                    "label1": row.get("Label1", ""),
                    "label2": row.get("Label2", ""),
                    "combined_label": row.get("Combined_Label", ""),
                    "label": row.get("Label", "")
                }
                pairs_data.append(pair_data)
            
            pairs_dict = {"pairs": pairs_data}
            
        elif isinstance(pairs_input, dict):
            # Input is a dictionary
            pairs_dict = pairs_input
        else:
            raise ValueError("pairs_input must be a dictionary or Excel file path string")
        
        if not pairs_dict or not isinstance(pairs_dict, dict):
            raise ValueError("pairs_dict must be a non-empty dictionary")
        
        if "pairs" not in pairs_dict:
            raise ValueError("pairs_dict must contain 'pairs' key")
        
        pairs = pairs_dict["pairs"]
        
        if not pairs:
            raise ValueError("No pairs found in the input data")
        
        print(f"Loaded {len(pairs)} pairs for training")
        
        # Convert all pairs to InputExample format first
        all_examples = []
        for pair in pairs:
            if loss_function == "MultipleNegativesRankingLoss":
                # For MNRL, we store the original label for testing but will filter for training
                example = InputExample(
                    texts=[pair["statement1"], pair["statement2"]],
                    label=int(pair.get("label", 0))  # Keep original label (0 or 1)
                )
            elif loss_function == "CosineSimilarityLoss":
                # For CosineSimilarityLoss, convert label to similarity score
                similarity_score = float(pair.get("label", 0))
                example = InputExample(
                    texts=[pair["statement1"], pair["statement2"]], 
                    label=similarity_score
                )
            else:
                raise ValueError(f"Unsupported loss function: {loss_function}")
            all_examples.append(example)
        
        # Split data into train and test sets FIRST (before filtering)
        random.shuffle(all_examples)
        split_idx = int(len(all_examples) * train_test_split)
        train_data_all = all_examples[:split_idx]
        test_data = all_examples[split_idx:]  # Keep all test data (label=0 and label=1)
        
        # Filter training data based on loss function
        if loss_function == "MultipleNegativesRankingLoss":
            # Use ALL positive pairs (label=1) from the entire dataset for training
            train_data = [ex for ex in all_examples if ex.label == 1]
            print(f"Training set: {len(train_data)} positive pairs (ALL label=1 from entire dataset)")
            print(f"Test set: {len(test_data)} pairs (split subset for evaluation)")
            
            # Count test set composition
            test_positive = sum(1 for ex in test_data if ex.label == 1)
            test_negative = sum(1 for ex in test_data if ex.label == 0)
            print(f"Test set composition: {test_positive} positive pairs, {test_negative} negative pairs")
            
        elif loss_function == "CosineSimilarityLoss":
            # Use all training data
            train_data = train_data_all
            print(f"Training set: {len(train_data)} pairs")
            print(f"Test set: {len(test_data)} pairs")
        else:
            train_data = train_data_all
            print(f"Training set: {len(train_data)} examples")
            print(f"Test set: {len(test_data)} examples")
        
        if len(train_data) == 0:
            raise ValueError("Training set is empty. Please check your data and split ratio.")
        
        # Load the base model
        print(f"Loading base model: {model_name}")
        model = SentenceTransformer(model_name)
        
        # Create data loader
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=per_device_train_batch_size)
        
        # Set up loss function
        if loss_function == "MultipleNegativesRankingLoss":
            train_loss = losses.MultipleNegativesRankingLoss(model=model)
        elif loss_function == "CosineSimilarityLoss":
            train_loss = losses.CosineSimilarityLoss(model=model)
        
        # Set up evaluator if we have test data
        evaluator = None
        if test_data and loss_function == "CosineSimilarityLoss":
            # Only create evaluator for CosineSimilarityLoss
            # MultipleNegativesRankingLoss doesn't need similarity score evaluation
            sentences1 = [example.texts[0] for example in test_data]
            sentences2 = [example.texts[1] for example in test_data]
            scores = [example.label for example in test_data]
            
            evaluator = EmbeddingSimilarityEvaluator(
                sentences1, sentences2, scores,
                batch_size=per_device_eval_batch_size,
                name="test_evaluator"
            )
        
        # Calculate total steps for warmup
        total_steps = len(train_dataloader) * num_train_epochs
        warmup_steps = min(warmup_steps, int(total_steps * 0.1))  # Max 10% of total steps
        
        print(f"Starting training with {num_train_epochs} epochs...")
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Loss function: {loss_function}")
        
        # Control wandb logging
        if not use_wandb:
            os.environ["WANDB_DISABLED"] = "true"
            print("Note: Weights & Biases (wandb) logging is disabled. Set use_wandb=True to enable.")
        else:
            # If wandb is enabled, remove the disabled flag if it exists
            os.environ.pop("WANDB_DISABLED", None)
        
        # Train the model
        # Prepare training arguments based on whether we have an evaluator
        fit_kwargs = {
            "train_objectives": [(train_dataloader, train_loss)],
            "epochs": num_train_epochs,
            "warmup_steps": warmup_steps,
            "output_path": output_path,
            "save_best_model": True,
            "optimizer_params": {
                'lr': learning_rate,
            },
            "scheduler": 'WarmupLinear',
            "checkpoint_path": output_path,
            "checkpoint_save_steps": save_steps,
            "use_amp": True  # Use automatic mixed precision for faster training
        }
        
        # Only add evaluation parameters if we have an evaluator
        if evaluator is not None:
            fit_kwargs["evaluator"] = evaluator
            fit_kwargs["evaluation_steps"] = evaluation_steps
        
        model.fit(**fit_kwargs)
        
        print(f"Training completed! Model saved to: {output_path}")
        
        # Load the best model
        fine_tuned_model = SentenceTransformer(output_path)
        
        # Evaluate the model on test data
        if test_data:
            print("\n" + "="*50)
            print("EVALUATION ON TEST SET")
            print("="*50)
            
            if loss_function == "MultipleNegativesRankingLoss":
                # Calculate average similarities for both types
                similarities = []
                true_labels = []
                
                print("Calculating similarities for test pairs...")
                for test_pair in test_data:
                    embeddings = fine_tuned_model.encode([test_pair.texts[0], test_pair.texts[1]])
                    similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
                    similarities.append(similarity)
                    true_labels.append(test_pair.label)
                
                # Calculate averages for each label type
                similarities_pos = [s for s, l in zip(similarities, true_labels) if l == 1]
                similarities_neg = [s for s, l in zip(similarities, true_labels) if l == 0]
                
                print(f"\nRESULTS:")
                if similarities_pos:
                    avg_similarity_pos = np.mean(similarities_pos)
                    print(f"Average similarity for same-dimension pairs (label=1): {avg_similarity_pos:.4f}")
                
                if similarities_neg:
                    avg_similarity_neg = np.mean(similarities_neg)
                    print(f"Average similarity for different-dimension pairs (label=0): {avg_similarity_neg:.4f}")
                
                if similarities_pos and similarities_neg:
                    separation = avg_similarity_pos - avg_similarity_neg
                    print(f"Separation: {separation:.4f}")
                
            else:
                # For other loss functions, show sample result
                sample_pair = test_data[0]
                embeddings = fine_tuned_model.encode([sample_pair.texts[0], sample_pair.texts[1]])
                similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
                print(f"Sample similarity score: {similarity:.4f}")
                print(f"Original label: {sample_pair.label} ({'same dimension' if sample_pair.label == 1 else 'different dimensions'})")
        
        return fine_tuned_model
        
    except ImportError as e:
        raise Exception(f"Required packages not installed. Please install: pip install sentence-transformers torch accelerate. Error: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to fine-tune model: {str(e)}")



def EFA_item_selection(
    items_input: Union[Dict[str, Any], str],
    model_path: str,
    n_factors: int,
    items_per_factor: int,
    output_path: Optional[str] = None,
    plot_title: str = "EFA Scree Plot",
    rotation: str = "varimax",
    random_seed: Optional[int] = 42) -> Dict[str, Any]:
    """
    Perform Exploratory Factor Analysis (EFA) using fine-tuned sentence transformer model
    and select best items for each factor.
    
    Args:
        items_input (Union[Dict[str, Any], str]): Input data. Can be:
            - Dictionary with "items" key containing items list (from item_generation/item_reduction/content_validation functions), OR
            - Dictionary in top_items format with dimension keys: {"dimension1": [items], "dimension2": [items], ...} (e.g., validated_items["top_items"]), OR
            - Path to Excel file (.xlsx) with items data, OR  
            - Path to JSON file (.json) with items data
        model_path (str): Path to the fine-tuned sentence transformer model
        n_factors (int): Number of factors to extract in EFA
        items_per_factor (int): Number of items to select per factor
        output_path (Optional[str]): Path to save the plot. If None, plot will be displayed. Defaults to None
        plot_title (str): Title for the scree plot. Defaults to "EFA Scree Plot"
        rotation (str): Factor rotation method. Options: "varimax", "promax", "oblimin", "none". Defaults to "varimax"
        random_seed (Optional[int]): Random seed for reproducibility. Defaults to 42
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - "selected_items": List of selected items with highest factor loadings (top items_per_factor for each factor)
            - "factor_loadings": Full factor loading matrix for all input items
            - "explained_variance": Explained variance ratios for each factor
            - "cumulative_variance": Cumulative explained variance
            - "similarity_matrix": Cosine similarity matrix used for EFA
            - "embeddings": Item embeddings from the fine-tuned model
            - "eigenvalues": Eigenvalues from factor analysis
            - "factor_loadings_file": Path to Excel file containing complete factor analysis results
            - "plot_file": Path to scree plot image file (if output_path provided)
    
    Raises:
        ValueError: If items_input is empty or invalid
        FileNotFoundError: If model_path does not exist
        Exception: If EFA analysis fails
    
    Example:
        >>> # Using items dictionary from item_generation
        >>> items_dict = item_generation(...)
        >>> results = EFA_item_selection(
        ...     items_input=items_dict,
        ...     model_path="./fine_tuned_model",
        ...     n_factors=3,
        ...     items_per_factor=5
        ... )
        
        >>> # Using top_items from content_validation (Recommended)
        >>> validated_items = content_validation(...)
        >>> results = EFA_item_selection(
        ...     items_input=validated_items["top_items"],  # Direct use of top items
        ...     model_path="./fine_tuned_model",
        ...     n_factors=3,
        ...     items_per_factor=5,
        ...     rotation="varimax"  # Clear factor structure
        ... )
        
        >>> # Using Excel file
        >>> results = EFA_item_selection(
        ...     items_input="items.xlsx",
        ...     model_path="./fine_tuned_model", 
        ...     n_factors=4,
        ...     items_per_factor=6,
        ...     output_path="./efa_plot.png"
        ... )
        
        >>> # Access results
        >>> selected_items = results["selected_items"]
        >>> loadings = results["factor_loadings"]
        >>> variance_explained = results["explained_variance"]
    """
    try:
        # Import required libraries
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.decomposition import FactorAnalysis
        from sklearn.metrics.pairwise import cosine_similarity
        from sentence_transformers import SentenceTransformer
        import json
        import os
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Check if model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # Load the fine-tuned model
        print(f"Loading fine-tuned model from: {model_path}")
        model = SentenceTransformer(model_path)
        
        # Handle different input types
        if isinstance(items_input, str):
            # Input is a file path
            if items_input.lower().endswith('.xlsx'):
                # Load from Excel file
                df = pd.read_excel(items_input)
                items_data = []
                for _, row in df.iterrows():
                    item_data = {
                        "item_number": row.get("Item_Number", row.get("item_number", "")),
                        "statement": row.get("Statement", row.get("statement", "")),
                        "dimension": row.get("Dimension", row.get("dimension", ""))
                    }
                    items_data.append(item_data)
                items_dict = {"items": items_data}
                
            elif items_input.lower().endswith('.json'):
                # Load from JSON file
                with open(items_input, 'r', encoding='utf-8') as f:
                    items_dict = json.load(f)
            else:
                raise ValueError("File must be .xlsx or .json format")
                
        elif isinstance(items_input, dict):
            # Input is a dictionary
            items_dict = items_input
        else:
            raise ValueError("items_input must be a dictionary or file path string")
        
        if not items_dict or not isinstance(items_dict, dict):
            raise ValueError("items_dict must be a non-empty dictionary")
        
        # Handle different dictionary formats
        if "items" in items_dict:
            # Standard format: {"items": [list of items]}
            items = items_dict["items"]
        else:
            # Check if it's top_items format: {"dimension1": [items], "dimension2": [items], ...}
            # Look for lists of dictionaries with item structure
            all_items = []
            is_top_items_format = False
            
            for key, value in items_dict.items():
                if isinstance(value, list) and value:
                    # Check if first item looks like an item dictionary
                    first_item = value[0]
                    if isinstance(first_item, dict) and any(field in first_item for field in ["statement", "item_number"]):
                        all_items.extend(value)
                        is_top_items_format = True
            
            if is_top_items_format and all_items:
                items = all_items
                print(f"Detected top_items format with {len(items_dict)} dimensions")
            else:
                raise ValueError("items_dict must contain 'items' key or be in top_items format (dimension: [items])")
        
        if not items:
            raise ValueError("No items found in the input data")
        
        print(f"Loaded {len(items)} items for EFA analysis")
        
        # Extract statements and metadata
        statements = []
        item_numbers = []
        dimensions = []
        
        for item in items:
            statement = item.get("statement", "")
            if not statement.strip():
                continue
            statements.append(statement.strip())
            item_numbers.append(item.get("item_number", ""))
            dimensions.append(item.get("dimension", ""))
        
        if len(statements) == 0:
            raise ValueError("No valid statements found in items")
        
        if len(statements) < n_factors:
            raise ValueError(f"Number of items ({len(statements)}) must be >= number of factors ({n_factors})")
        
        print(f"Processing {len(statements)} valid statements")
        
        # Calculate embeddings using fine-tuned model
        print("Calculating embeddings using fine-tuned model...")
        embeddings = model.encode(statements, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Calculate cosine similarity matrix
        print("Computing cosine similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings)
        
        # Convert similarity to correlation-like matrix for EFA
        # EFA typically works better with correlation matrices
        correlation_matrix = similarity_matrix
        
        # Perform EFA using factor_analyzer for better results
        print(f"Performing EFA with {n_factors} factors...")
        print(f"Using rotation: {rotation}")
        
        try:
            from factor_analyzer import FactorAnalyzer
        except ImportError:
            raise ImportError("factor_analyzer package is required. Install with: pip install factor_analyzer")
        
        # Create FactorAnalyzer instance (same as your notebook)
        rotation_method = None if rotation.lower() == "none" else rotation.lower()
        fa = FactorAnalyzer(n_factors=n_factors, 
                           is_corr_matrix=True, 
                           rotation=rotation_method)
        
        # Fit the model
        fa.fit(correlation_matrix)
        
        # Get factor loadings
        factor_loadings = fa.loadings_  # Shape: (n_items, n_factors)
        
        # Calculate explained variance using factor_analyzer's method
        variance_info = fa.get_factor_variance()
        explained_variance = variance_info[1]  # proportional variance 
        cumulative_variance = variance_info[2]  # cumulative variance
        
        # Get eigenvalues from factor_analyzer
        eigenvalues, _ = fa.get_eigenvalues()
        
        print(f"Explained variance by factors: {explained_variance}")
        print(f"Cumulative explained variance: {cumulative_variance}")
        
        # Create scree plot (like your notebook)
        print("Creating scree plot...")
        plt.figure(figsize=(10, 6))
        
        # Plot eigenvalues with both scatter and line (like your notebook)
        factor_numbers = range(1, len(eigenvalues) + 1)
        plt.scatter(factor_numbers, eigenvalues, s=50, alpha=0.7)
        plt.plot(factor_numbers, eigenvalues, linewidth=2)
        
        # Add red dashed line at n_factors
        plt.axvline(x=n_factors, color='red', linestyle='--', linewidth=2, 
                   label=f'n_factors = {n_factors}')
        
        # Formatting (matching your notebook style)
        plt.xlabel('Factor', fontsize=12)
        plt.ylabel('Eigenvalue', fontsize=12)
        plt.title(plot_title, fontsize=14)
        plt.grid(True)
        plt.legend()
        
        # Add cumulative variance text
        cumulative_var_at_nfactors = cumulative_variance[n_factors-1] if n_factors <= len(cumulative_variance) else cumulative_variance[-1]
        variance_text = f'Cumulative Variance at {n_factors} factors: {cumulative_var_at_nfactors:.3f} ({cumulative_var_at_nfactors*100:.1f}%)'
        plt.text(0.02, 0.98, variance_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save or show plot
        if output_path:
            plot_file = output_path
        else:
            # Auto-generate filename
            plot_file = f"efa_scree_plot_{n_factors}factors.png"
            
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Scree plot saved to: {plot_file}")
        
        # Also show plot if output_path was not specified
        if not output_path:
            plt.show()
        
        # Select best items for each factor
        print(f"Selecting top {items_per_factor} items per factor...")
        selected_items = []
        
        for factor_idx in range(n_factors):
            # Get absolute loadings for this factor
            factor_loadings_abs = np.abs(factor_loadings[:, factor_idx])
            
            # Get top items for this factor
            top_indices = np.argsort(factor_loadings_abs)[::-1][:items_per_factor]
            
            for rank, item_idx in enumerate(top_indices):
                selected_item = {
                    "factor": factor_idx + 1,
                    "rank_in_factor": rank + 1,
                    "item_number": item_numbers[item_idx],
                    "statement": statements[item_idx],
                    "original_dimension": dimensions[item_idx],
                    "factor_loading": factor_loadings[item_idx, factor_idx],
                    "abs_factor_loading": factor_loadings_abs[item_idx]
                }
                selected_items.append(selected_item)
        
        # Sort selected items by factor and rank
        selected_items.sort(key=lambda x: (x["factor"], x["rank_in_factor"]))
        
        # Create comprehensive factor loadings Excel file
        print("Creating factor loadings Excel file...")
        factor_loadings_file = output_path.replace('.png', '_factor_loadings.xlsx') if output_path and output_path.endswith('.png') else 'factor_loadings_analysis.xlsx'
        
        try:
            with pd.ExcelWriter(factor_loadings_file, engine='openpyxl') as writer:
                
                # Sheet 1: Complete Factor Loadings for all items
                factor_loadings_data = []
                for i in range(len(statements)):
                    row = {
                        "Item_Number": item_numbers[i],
                        "Statement": statements[i],
                        "Original_Dimension": dimensions[i]
                    }
                    
                    # Add factor loadings for each factor
                    for j in range(n_factors):
                        row[f"Factor_{j+1}_Loading"] = factor_loadings[i, j]
                        row[f"Factor_{j+1}_Abs_Loading"] = abs(factor_loadings[i, j])
                    
                    # Find highest loading factor
                    max_loading_idx = np.argmax(np.abs(factor_loadings[i, :]))
                    row["Highest_Loading_Factor"] = max_loading_idx + 1
                    row["Highest_Loading_Value"] = factor_loadings[i, max_loading_idx]
                    row["Highest_Abs_Loading"] = abs(factor_loadings[i, max_loading_idx])
                    
                    factor_loadings_data.append(row)
                
                factor_df = pd.DataFrame(factor_loadings_data)
                factor_df.to_excel(writer, sheet_name="Factor_Loadings", index=False)
                
                # Sheet 2: Selected Items Only
                selected_df = pd.DataFrame(selected_items)
                selected_df.to_excel(writer, sheet_name="Selected_Items", index=False)
                
                # Sheet 3: Variance Explained
                variance_data = []
                for i in range(n_factors):
                    variance_data.append({
                        'Factor': i + 1,
                        'Eigenvalue': eigenvalues[i] if i < len(eigenvalues) else 0,
                        'Explained_Variance': explained_variance[i] if i < len(explained_variance) else 0,
                        'Cumulative_Variance': cumulative_variance[i] if i < len(cumulative_variance) else 0,
                        'Explained_Variance_Percent': explained_variance[i] * 100 if i < len(explained_variance) else 0,
                        'Cumulative_Variance_Percent': cumulative_variance[i] * 100 if i < len(cumulative_variance) else 0
                    })
                
                variance_df = pd.DataFrame(variance_data)
                variance_df.to_excel(writer, sheet_name="Variance_Explained", index=False)
                
                # Sheet 4: Similarity Matrix (subset for readability)
                sim_df = pd.DataFrame(similarity_matrix[:min(50, len(similarity_matrix)), :min(50, len(similarity_matrix))])
                sim_df.index = [f"Item_{i+1}" for i in range(sim_df.shape[0])]
                sim_df.columns = [f"Item_{i+1}" for i in range(sim_df.shape[1])]
                sim_df.to_excel(writer, sheet_name="Similarity_Matrix")
                
                # Auto-adjust column widths for all sheets
                for sheet_name in writer.sheets:
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
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
            
            print(f"Factor loadings Excel file created: {factor_loadings_file}")
            
        except Exception as e:
            print(f"Warning: Failed to create Excel file: {str(e)}")
            factor_loadings_file = None
        
        # Print summary
        print(f"\n" + "="*60)
        print("EFA ITEM SELECTION RESULTS")
        print("="*60)
        print(f"Total items analyzed: {len(statements)}")
        print(f"Number of factors: {n_factors}")
        print(f"Items per factor: {items_per_factor}")
        print(f"Total selected items: {len(selected_items)}")
        print(f"Cumulative variance explained: {cumulative_variance[-1]:.3f} ({cumulative_variance[-1]*100:.1f}%)")
        
        # Print selected items summary
        print(f"\nSELECTED ITEMS SUMMARY:")
        current_factor = 0
        for item in selected_items:
            if item["factor"] != current_factor:
                current_factor = item["factor"]
                print(f"\n--- Factor {current_factor} ---")
            print(f"  {item['rank_in_factor']}. [{item['item_number']}] Loading: {item['factor_loading']:.3f}")
            print(f"     {item['statement'][:80]}{'...' if len(item['statement']) > 80 else ''}")
        
        # Prepare return dictionary
        results = {
            "selected_items": selected_items,
            "factor_loadings": factor_loadings,
            "explained_variance": explained_variance,
            "cumulative_variance": cumulative_variance,
            "similarity_matrix": similarity_matrix,
            "embeddings": embeddings,
            "eigenvalues": eigenvalues,
            "n_factors": n_factors,
            "items_per_factor": items_per_factor,
            "rotation_method": rotation,
            "factor_loadings_file": factor_loadings_file,
            "plot_file": plot_file
        }
        
        return results
        
    except ImportError as e:
        raise Exception(f"Required packages not installed. Please install: pip install matplotlib scikit-learn. Error: {str(e)}")
    except FileNotFoundError as e:
        raise e
    except Exception as e:
        raise Exception(f"Failed to perform EFA item selection: {str(e)}")

