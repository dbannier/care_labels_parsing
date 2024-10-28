"""
File defining Pydantic models for care_label data
"""
from pydantic import BaseModel
from typing import Optional

class Category(BaseModel):
    product_main_category:str 
    product_sub_category: Optional[str] = None

class Color(BaseModel):
    color: Optional[str] = None

class Component(BaseModel):
    component_name: str 
    composition: dict
    additional_details: Optional[str] = None
    weight: Optional[float] = None

class ProductDetails(BaseModel):
    product_id: str
    category: Category 
    color: Optional[Color] = None 
    component: Component