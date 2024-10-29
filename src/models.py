"""
File defining Pydantic models for care_label data
"""
from typing import Optional

from pydantic import BaseModel


class Category(BaseModel):
    """Model defining catergory and subcategory of a product."""
    product_main_category:str
    product_sub_category: Optional[str] = None

class Color(BaseModel):
    """Model defining color of a product, if any."""
    color: Optional[str] = None

class Component(BaseModel):
    """Model giving details about the component and material."""
    component_name: str
    composition: dict
    additional_details: Optional[str] = None
    weight: Optional[float] = None

class ProductDetails(BaseModel):
    """Global model providing all information for a product."""
    product_id: str
    category: Category
    color: Optional[Color] = None
    component: Component
