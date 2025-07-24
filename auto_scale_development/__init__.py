"""
Auto Scale Development Package

A Python package for automated scale development and analysis.
"""

__version__ = "0.1.0"
__author__ = "Auto Scale Development Team"

from .config import get_api_key, call_llm_api, get_model_provider
from .core_functions import (
    item_generation, 
    item_reduction,
    content_validation,
    export_items_to_excel, 
    export_items_to_json,
    get_items
)
# Note: Helper functions are internal and not exposed in __all__
# They are used internally by core functions

__all__ = [
    "get_api_key", 
    "call_llm_api", 
    "get_model_provider", 
    "item_generation", 
    "item_reduction",
    "content_validation",
    "export_items_to_excel", 
    "export_items_to_json",
    "get_items"
] 