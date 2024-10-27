# File containing functions to process care_label file

import json
import re

import pandas as pd
from nltk.corpus import stopwords
from pydantic import BaseModel, ValidationError

from src.models import Category, Color, Component, ProductDetails


def strip_and_trim_punctuation(description_series: pd.Series) -> pd.Series:
    """ Function to stip pandas series string input and trim ending punctuation."""
    punctuation = ".,;:!?" #Punctuation to remove at the end of the rows
    return description_series.str.strip().str.rstrip(punctuation).str.lstrip(punctuation)


def lower_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Function lowering text for all the input dataframe"""
    return df.applymap(lambda x: x.lower() if pd.notnull(x) else x)


def remove_english_stopwords(description_series:pd.Series)-> pd.Series:
    """ Function removing enflish stopwords from input text series"""
    try:
        # Try to load stopwords
        sw = stopwords.words("english")
    except LookupError:
        # Download if missing
        import nltk
        nltk.download("stopwords")

    return description_series.apply(lambda x: " ".join([word for word in x.split() if word not in sw]))

        
def replace_and_comma(description_series : pd.Series)-> pd.Series : 
    """Function replacing 'and' words with commas from input text series"""
    return description_series.str.replace(" and ", ", ", regex=True)


def remove_escape_characters(descriptive_series:pd.Series)->pd.Series:
    """Function removing escape characters ("\n") from input series"""
    return descriptive_series.str.replace("\n", " ", regex=True)

def remove_commas(descriptive_series:pd.Series)->pd.Series:
    """Function removing escape characters ("\n") from input series"""
    return descriptive_series.str.replace(",", " ", regex=True)

def remove_symbols(description_series: pd.Series):
    """Function removing symbols from input text series"""
    symbols_to_remove = ["®", "+", "™","/"]
    # Create a regex pattern that matches any of the symbols
    pattern = '|'.join(map(re.escape, symbols_to_remove))
    
    # Replace all matched symbols with an empty string
    return description_series.str.replace(pattern, '', regex=True)


def replace_words(description_series: pd.Series, list_to_replace: list[str], replacement_word: str) -> pd.Series:
    """
    Replace each word or phrase in `list_to_replace` with `replacement_word` in a given Series.

    Parameters:
    - description_series (pd.Series): The Series to perform replacements on.
    - list_to_replace (list[str]): A list of words or phrases to replace.
    - replacement_word (str): The word to use as the replacement.

    Returns:
    - pd.Series: The modified Series with specified replacements applied.
    """
    for word in list_to_replace:
        description_series = description_series.replace(word, replacement_word, regex=True)
    description_series = strip_and_trim_punctuation(description_series)
    return description_series.str.strip()


def split_colors(df: pd.DataFrame, description_column:str) -> pd.DataFrame:
    """
    Function creating a new row for each different composition of an article based on its color. 
    A color column includes the color information for the given row.
    Parameters: 
    - df: dataframe input
    - description_column: column name to transform
    Returns: the dataframe with a new color column and the specified column transformed so that it is specific to one set of colors.
    """
    color_group_pattern = r'colour\b[\s\.,;:-]+(\d{4}(?:[\s,]+\d{4})*(?:\s*and\s*\d{4})?)' #looks for  "colour" + following colors        
    color_pattern = r'\bcol(?:\.|ours?|ors?)?\b' # look for a color indicator
    punct_cleanup_pattern = r'^[\.,;:!?-]+|[\.,;:!?-]+$' # punctuation to remove

    rows = []
    
    for _, row in df.iterrows():
        # Split composition by group of colors with the same composition
        compositions = re.split(f'(?={color_pattern})', row[description_column])
        
        for composition in filter(None, map(str.strip, compositions)):
            # Replace color indicator by unique indicator for easier spotting
            composition = re.sub(color_pattern, "colour", composition)

            # Spot all color groups
            color_groups = re.findall(color_group_pattern, composition)

            # Remove the colors indication from the care_label field
            updated_composition = re.sub(color_group_pattern, "", composition).strip()

            # Remove punctuation at the end of the description for standardization
            updated_composition = re.sub(punct_cleanup_pattern, "", updated_composition).strip()

            if color_groups:
                for color in color_groups:
                    # Add a row for each color group
                    rows.append({**row, description_column: updated_composition, "color": color})
            else:
                rows.append({**row, description_column: composition, "color": None})

    return pd.DataFrame(rows)


def split_sentence(df: pd.DataFrame,description_column:str) -> pd.DataFrame:
    """
    Function creating one row per sentence for the specified column.
    A sentence is defined by 
    - a dot between a digit and a word 
    - a dot between a word and a digit 
    - a dot between two words
    - the hit of 'gsm' word
    Parameters : 
    - df :the dataframe 
    - description_column: the column of the dataframe to transform
    Returns: the transformed dataframe
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Regex defining end of a sentence
    sentence_endings = re.compile(r'(?<=\d)\.(?=\D)|(?<=\D)\.(?=\d)|(?<=\D)\.(?=\D)|(?<=gsm)')

    # Split the specified column by sentence
    df_copy[description_column] = df_copy[description_column].apply(lambda x: sentence_endings.split(x))

    # Create one row per "sentence"
    df_copy = df_copy.explode(description_column)
    df_copy[description_column] = strip_and_trim_punctuation(df_copy[description_column])

    # Remove empty rows
    df_copy = df_copy[(df_copy[description_column] != "") & (df_copy[description_column] != ".")]
    return df_copy.reset_index(drop=True)


def split_components(df: pd.DataFrame,description_column: str):
    """
    Splits components from the 'updated_care_label' column of a DataFrame.

    The function searches for components defined in the format "Component Name: Composition" 
    within the 'updated_care_label' column value. If no components are found, it assigns
    the component name "main".
    Parameters : 
    - df : pandas.DataFrame
    - description_column: the column to update
    Returns: pandas.DataFrame with a new column 'component' containing the name of the component
    extracted from the 'updatedcare_label' (or "main" if none found).
    """
    # Regex defining the component name and its composition :
    # - words before a colon
    # - everything that comes after the colon for the composition
    component_pattern = r'([a-zA-Z\s]+):\s*(.*?)(?=\s*[a-zA-Z\s]+:|$)'
    
    rows = []
    for _, row in df.iterrows():

        # find components or if there is none, call the component "main"
        components = re.findall(component_pattern, row[description_column]) or [("main", row[description_column])]

        for component_name, component_comp in components:
            # Add a row for each component and its composition
            rows.append({
                **row,
                "component": component_name.strip(),
                description_column: component_comp,
            })
    result_df = pd.DataFrame(rows)
    result_df.updated_care_label = strip_and_trim_punctuation(result_df.updated_care_label )
    return result_df.reset_index(drop=True)


def get_weight(df: pd.DataFrame,description_column: str) -> pd.DataFrame:
    """
    Function extracting weight information using 'gsm' to spot its location.
    Parameters : 
    - df : pandas.DataFrame
    - description_column: the column to update
    Returns : the dataframe with a new column containing the weight information in g/m2.
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Extract digits preceding "gsm" word
    weights = df_copy[description_column].str.split(r'(\d+)\s*,?\s*gsm',expand=True)
    df_copy[[description_column,"weight"]] = weights[[0,1]]

    # Clean the description column for extra spaces and ending punctuation
    df_copy[description_column] = strip_and_trim_punctuation(df_copy[description_column])
    return df_copy

def dataframe_to_pydantic(df: pd.DataFrame, model: BaseModel):
    """
    Convert a pandas DataFrame to a list of Pydantic models.
    
    Parameters:
    - df: The input DataFrame.
    - model: The Pydantic model to instantiate.
    
    Returns:
    - A list of instantiated Pydantic models.
    """
    instances = []
    for _, row in df.iterrows():
        try:
            instance = model(
                product_id=row.get('product_id'),  # Use get to avoid KeyErrors
                category=Category(
                    product_main_category=row.get('product_main_category'),
                    product_sub_category=row.get('product_sub_category')
                ),
                color=Color(color=row.get('color')),  # This will be None if color is missing
                component=Component(
                    component=row.get('component'),
                    updated_care_label=row.get('updated_care_label'),
                    weight=row.get('weight')
                )
            )
            instances.append(instance)
        except ValidationError as e:
            print(f"Validation error for row {row}: {e}")
            # Handle or log the error as needed

    return instances

def pydanticlist_to_json(pydanticlist: list, file_name: str) -> None:
    """ Function converting a list of models into a json file """
    items_json = [item.dict() for item in pydanticlist]  # Using dict() instead of json() to get a native Python dict

    # Remove json extension if any 
    file_name = re.sub(r"\.json$", "", file_name)

    # Save to a JSON file
    with open(f'{file_name}.json', 'w') as json_file:
        json.dump(items_json, json_file, indent=4)
