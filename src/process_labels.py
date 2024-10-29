""" File containing functions to process care_label file """

import json
import logging
import re
import sys

import pandas as pd
import nltk
from nltk.corpus import stopwords
from pydantic import ValidationError

from src.models import Category, Color, Component, ProductDetails


# Constants
# ways of writing gram per square meter - to replace by "gsm"
ALL_GSM = [
    "g/m2",
    "g/m²",
    "gm²",
    "gm2",
    " gram.",
    " gram ",
    "gr ",
    "gr.",
    " g ",
]

PUNCTUATION = ".,;:!?"


def strip_and_trim_punctuation(description_series: pd.Series) -> pd.Series:
    """ Function to stip pandas series string input and trim ending punctuation."""
     #Punctuation to remove at the end of the rows
    return description_series.str.strip().str.rstrip(PUNCTUATION).str.lstrip(PUNCTUATION)

def preprocess_series(description_series: pd.Series)-> pd.Series:
    """
    Standard function to preprocess a pandas series containing text.

    Parameters:
    - a pandas series, containing string 
    Returns:
    the processed pandas series
    """
    # Replace "and" by comma
    description_series = description_series.str.replace(" and ", ", ", regex=True)

    # Remove stopwords
    description_series = remove_english_stopwords(description_series)

    # Remove escape characters
    description_series = description_series.str.replace("\n", " ", regex=True)

    # Remove symbols
    description_series = remove_symbols(description_series)
    return description_series


def remove_english_stopwords(description_series:pd.Series)-> pd.Series:
    """ Function removing english stopwords from input text series"""
    try:
        # Try to load stopwords
        sw = stopwords.words("english")
    except LookupError:
        # Download if missing
        nltk.download("stopwords")

    return description_series.apply(
        lambda x: " ".join(
            [word for word in x.split() if word not in sw]
        )
    )


def remove_symbols(description_series: pd.Series):
    """Function removing symbols from input text series"""
    symbols_to_remove = ["®", "+", "™","/"]
    # Create a regex pattern that matches any of the symbols
    pattern = "|".join(map(re.escape, symbols_to_remove))

    # Replace all matched symbols with an empty string
    return description_series.str.replace(pattern, "", regex=True)


def replace_words(
    description_series: pd.Series,
    list_to_replace: list[str],
    replacement_word: str
    ) -> pd.Series:
    """
    Replace each word in `list_to_replace` with `replacement_word` in a given Series.

    Parameters:
    - description_series (pd.Series): The Series to perform replacements on.
    - list_to_replace (list[str]): A list of words to replace.
    - replacement_word (str): The word to use as the replacement.

    Returns:
    - pd.Series: The modified Series with specified replacements applied.
    """
    for word in list_to_replace:
        pattern = re.escape(word) + r"\b"
        description_series = description_series.replace(pattern, replacement_word, regex=True)
    description_series = strip_and_trim_punctuation(description_series)
    return description_series.str.strip()


def split_colors(df: pd.DataFrame, description_column:str) -> pd.DataFrame:
    """
    Function creating a new row for each different composition of an article based on its color. 
    A color column includes the color information for the given row with an identifier of a color
    (e.g "col", "colour") followed by one or multiple color ids made up of 4 digits.
    Parameters: 
    - df: dataframe input
    - description_column: column name to transform
    Returns: the dataframe with a new color column and the specified column transformed
            so that it is specific to one set of colors.
    """
    #looks for  "colour" + following colors
    color_group_pattern = r"colour\b[\s\.,;:-]+(\d{4}(?:[\s,]+\d{4})*(?:\s*and\s*\d{4})?)"
    color_pattern = r"\bcol(?:\.|ours?|ors?)?\b" # look for a color indicator
    punct_cleanup_pattern = r"^[\.,;:!?-]+|[\.,;:!?-]+$" # punctuation to remove

    rows = []

    for _, row in df.iterrows():
        # Split composition by group of colors with the same composition
        compositions = re.split(f"(?={color_pattern})", row[description_column])

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
    - the hit of "gsm" word
    Parameters : 
    - df :the dataframe 
    - description_column: the column of the dataframe to transform
    Returns: the transformed dataframe
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Regex defining end of a sentence
    sentence_endings = re.compile(r"(?<=\d)\.(?=\D)|(?<=\D)\.(?=\d)|(?<=\D)\.(?=\D)|(?<=gsm)")

    # Split the specified column by sentence
    df_copy[description_column] = df_copy[description_column].apply(
        lambda x: sentence_endings.split(x)
    )

    # Create one row per "sentence"
    df_copy = df_copy.explode(description_column)
    df_copy[description_column] = strip_and_trim_punctuation(df_copy[description_column])

    # Remove empty rows
    df_copy = df_copy[(df_copy[description_column] != "") & (df_copy[description_column] != ".")]
    return df_copy.reset_index(drop=True)


def split_components(df: pd.DataFrame,description_column: str):
    """
    Splits components from the "updated_care_label" column of a DataFrame.

    The function searches for components defined in the format "Component Name: Composition" 
    within the "updated_care_label" column value. If no components are found, it assigns
    the component name "main".
    The extracted information is removed from the input column of the dataframe.
    Parameters : 
    - df : pandas.DataFrame
    - description_column: the column to update
    Returns: pandas.DataFrame with a new column "component" containing the name of the component
    extracted from the "updated_care_label" (or "main" if none found).
    """
    # Regex defining the component name and its composition :
    # - words before a colon
    # - everything that comes after the colon for the composition
    component_pattern = r"([a-zA-Z\s]+):\s*(.*?)(?=\s*[a-zA-Z\s]+:|$)"

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
    Function extracting weight information using "gsm" to spot its location.
    The extracted information is removed from the input column of the dataframe.
    Parameters : 
    - df : pandas.DataFrame
    - description_column: the column to update
    Returns : the dataframe with a new column containing the weight information in g/m2 (gsm).
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Extract digits preceding "gsm" word
    weights = df_copy[description_column].str.split(r"(\d+)\s*,?\s*gsm",expand=True)
    df_copy[[description_column,"weight"]] = weights[[0,1]]

    # Clean the description column for extra spaces and ending punctuation
    df_copy[description_column] = strip_and_trim_punctuation(df_copy[description_column])
    return df_copy


def parse_composition(composition_text:str)-> (str,dict):
    """
    Function separating the material name and its percentage.
    N.B : This only works well if the percentage is given before the material name.
    The extracted information is removed from the input column of the dataframe.
    Ex : 50% cotton, 10% polyamide -> {cotton : 50, polyamide : 10}
    Parameters : 
     - composition_text : string containing the percentage of a given specified material
    Returns : 
    - the input string cleaned from the extracted information
    - the disctionary containing the percentage for each material
    """
    # Regex to capture percentage and material pairs
    pattern = r"(\d+(?:[.,]\d+)?)\s*%\s*([^%]*?)(?=\d+(?:[.,]\d+)?\s*%|$)"

    # Find all matches in the composition text
    matches = re.findall(pattern, composition_text)

    # Convert matches to a dictionary: {material: percentage}
    composition_dict = {
    material.strip(): float(percentage.replace(",", ".")) if percentage else None
    for percentage, material in matches
    }

    # Remove matched text from description
    cleaned_text = re.sub(pattern, "", composition_text).strip()
    return cleaned_text, composition_dict


def dataframe_to_pydantic(df: pd.DataFrame)->list:
    """
    Convert a pandas DataFrame to a list of Pydantic models.
    
    Parameters:
    - df: The input DataFrame.
   
    Returns:
    - A list of instantiated Pydantic models.
    """
    instances = []
    for _, row in df.iterrows():
        try:
            instance = ProductDetails(
                product_id=row.get("product_id"),  # Use get to avoid KeyErrors
                category=Category(
                    product_main_category=row.get("product_main_category"),
                    product_sub_category=row.get("product_sub_category")
                ),
                color=Color(color=row.get("color")),  # This will be None if color is missing
                component=Component(
                    component_name=row.get("component"),
                    composition=row.get("composition_dict"),
                    additional_details=row.get("remaining_text"),
                    weight=row.get("weight")
                )
            )
            instances.append(instance)
        except ValidationError as e:
            print(f"Validation error for row {row}: {e}")
            # Handle or log the error as needed

    return instances

def pydanticlist_to_json(pydanticlist: list, file_name: str) -> None:
    """ Function converting a list of models into a json file """

    items_json = [item.dict() for item in pydanticlist]

    # Remove json extension if any
    file_name = re.sub(r"\.json$", "", file_name)

    # Save to a JSON file
    with open(f"{file_name}.json", "w") as json_file:
        json.dump(items_json, json_file, indent=4)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Input file missing ! Usage: python script.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    # Set logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Read file
    logging.info(f"Reading file from {input_file}")
    label_file = pd.read_csv(input_file)

    #### Preprocessing ####
    logging.info("Preprocessing file...")
    # Lower dataframe
    clean_label_file = label_file.applymap(lambda x: x.lower() if pd.notnull(x) else x)

    # Split categories to get main category and subcategory field
    clean_label_file[["product_main_category","product_sub_category"]]= clean_label_file.product_category.str.split("/",expand= True)

    # Preprocess care label series
    clean_label_file["updated_care_label"] = preprocess_series(clean_label_file.care_label)

    # Standardize unit of measure
    clean_label_file.updated_care_label = replace_words(clean_label_file.updated_care_label, ALL_GSM, "gsm")


    #### Parsing ####
    logging.info("Parsing file...")
    # Color parsing
    clean_label_file_colors = split_colors(
        clean_label_file,
        "updated_care_label",
    )

    ##Create one row per component of each item
    clean_label_file_item = split_sentence(
        clean_label_file_colors,
        "updated_care_label",
    )

    # Extract component name
    clean_label_file_component = split_components(
        clean_label_file_item,
        "updated_care_label",
    )

    # Extract weight information from the updated care label column
    clean_label_file_weight = get_weight(
        clean_label_file_component,
        "updated_care_label",
    )

    # Extract composition details
    clean_label_file_weight[["remaining_text", "composition_dict"]] = clean_label_file_weight["updated_care_label"].apply(
        lambda x: pd.Series(parse_composition(x))
    )

    # Clean remaining text
    clean_label_file_weight["remaining_text"] = clean_label_file_weight["remaining_text"].str.replace(",", " ", regex=True)
    clean_label_file_weight["remaining_text"] = strip_and_trim_punctuation(
        clean_label_file_weight["remaining_text"]
    )

    # Set weight data type
    clean_label_file_weight.weight=clean_label_file_weight.weight.astype(float)

    logging.info("Saving results file...")
    # Products with remaining text may need special attention
    clean_label_file_weight[clean_label_file_weight["remaining_text"]!=""].to_excel("data/processed/to_review.xlsx")

    # Transform dataframe to structured json and save results
    clean_label_file_weight.to_csv("data/processed/final_care_label.csv")
    products = dataframe_to_pydantic(clean_label_file_weight)
    pydanticlist_to_json(products,"data/processed/products_database")

    logging.info("Done ! ")
