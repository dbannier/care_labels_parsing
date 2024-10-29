"""
File containing unit test functions for process_labels.py
"""
import json
import nltk
import pandas as pd
import pytest

from src.models import Category, Color, Component, ProductDetails
from src.process_labels import (
    dataframe_to_pydantic,
    get_weight,
    parse_composition,
    pydanticlist_to_json,
    remove_english_stopwords,
    remove_symbols,
    replace_words,
    split_colors,
    split_components,
    split_sentence,
    strip_and_trim_punctuation,
)


def test_strip_and_trim_punctuation():
    data = pd.Series([" Hello, world! ", "Test; string.", "No punctuation"])
    expected = pd.Series(["Hello, world", "Test; string", "No punctuation"])
    result = strip_and_trim_punctuation(data)
    pd.testing.assert_series_equal(result, expected)


def test_remove_english_stopwords():
    nltk.download("stopwords")
    data = pd.Series(["this is a test string", "another example with stopwords"])
    expected = pd.Series(["test string", "another example stopwords"])
    result = remove_english_stopwords(data)
    pd.testing.assert_series_equal(result, expected)


def test_remove_symbols():
    data = pd.Series(["This is a test®", "Remove + symbols™", "Slash / symbol"])
    expected = pd.Series(["This is a test", "Remove  symbols", "Slash  symbol"])
    result = remove_symbols(data)
    pd.testing.assert_series_equal(result, expected)


def test_replace_words():
    # make sure to match words stick to numbers
    data = pd.Series(["This is a test sentence.", "Another test case.", "No match here."])
    list_to_replace = ["test", "case"]
    replacement_word = "example"
    expected = pd.Series(["This is a example sentence", "Another example example", "No match here"])
    result = replace_words(data, list_to_replace, replacement_word)
    pd.testing.assert_series_equal(result, expected)


def test_split_colors():
    ###### WIP
    # make sure to replace and by commas in color column
    data = pd.DataFrame({
        "description": [
            "colour 1234 and 5678: This is a product with information. colors 1234 other info",
            "col. 9101, 1121 Another product",
            "Product without information."
        ],
        "other_column": ["value1", "value2", "value3"]
    })
    
    expected = pd.DataFrame([
        {"description": "This is a product with information", "other_column": "value1", "color": "1234 and 5678"},
        {"description": "other info", "other_column": "value1", "color": "1234"},
        {"description": "Another product", "other_column": "value2", "color": "9101, 1121"},
        {"description": "Product without information.", "other_column": "value3", "color": None}
    ], index = [0,1,2,3])
    
    result = split_colors(data, "description")
    pd.testing.assert_frame_equal(result, expected)


def test_split_sentence():
    data = pd.DataFrame({
        "description": [
            "This is a test. Another sentence. 123.456 gsm",
            "Sentence with a number 789. Another one.",
            "No special cases here."
        ],
        "other_column": ["value1", "value2", "value3"]
    })
    
    expected = pd.DataFrame([
        {"description": "This is a test", "other_column": "value1"},
        {"description": "Another sentence", "other_column": "value1"},
        {"description": "123.456 gsm", "other_column": "value1"},
        {"description": "Sentence with a number 789", "other_column": "value2"},
        {"description": "Another one", "other_column": "value2"},
        {"description": "No special cases here", "other_column": "value3"}
    ])
    
    result = split_sentence(data, "description")
    pd.testing.assert_frame_equal(result, expected)

def test_split_components():
    # There should be only one component per row. 
    # data has been lowered already
    #### to improve
    # Sample data

    data = {
        'updated_care_label': [
            'body: 10% cotton 90% polyester.',
            '100% wool',
            'lining: 50% silk 50% cotton'
        ]
    }
    df = pd.DataFrame(data)

    # Expected output
    expected_data = {
        'updated_care_label': [
            '10% cotton 90% polyester',
            '100% wool',
            '50% silk 50% cotton',
        ],
        'component': [
            'body',
            'main',
            'lining',
        ]
    }
    expected_df = pd.DataFrame(expected_data)

    # Run the function
    result_df = split_components(df, 'updated_care_label')
    # Check if the result matches the expected output
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_get_weight():
    """
    The functions tests following cases  : 
    - weight + space + gsm
    - weight + gsm
    - weight + comma + space + gsm
    - no gsm
    """
    # Sample data
    data = {
        'updated_care_label': [
            'fabric: 200 gsm',
            'material 150gsm',
            'weight: 300, gsm',
            'no weight info'
        ]
    }
    df = pd.DataFrame(data)

    # Expected output
    expected_data = {
        'updated_care_label': [
            'fabric',
            'material',
            'weight',
            'no weight info'
        ],
        'weight': [
            '200',
            '150',
            '300',
            None
        ]
    }
    expected_df = pd.DataFrame(expected_data)

    # Run the function
    result_df = get_weight(df, 'updated_care_label')

    # Check if the result matches the expected output
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_parse_composition():
    """
    tested use cases : 
    - two materials separated by a comma
    - one material
    - 2 materials separated by a space 
    - percentage in float format
    - material containing comma and hyphen

    """
    #####need to clean end punctuation of materials 

    # Test cases
    test_cases = [
        ("50% cotton, 10% polyamide", ("", {"cotton,": 50.0, "polyamide": 10.0})),
        ("100% Wool", ("", {"Wool": 100.0})),
        ("80% Cotton 20% Polyester", ("", {"Cotton": 80.0, "Polyester": 20.0})),
        ("No percentages here", ("No percentages here", {})),
        ("50.5% 37,5-Silk, 49.5% Linen", ("", {"37,5-Silk,": 50.5, "Linen": 49.5}))
    ]

    # Run tests
    for i, (input_text, expected_output) in enumerate(test_cases):
        result = parse_composition(input_text)
        assert result == expected_output, f"Test case {i+1} failed: expected {expected_output}, got {result}"


def test_dataframe_to_pydantic():
    """
    Test function for dataframe_to_pydantic to ensure it correctly converts a DataFrame to a list of Pydantic models.
    """
    # Sample data for the DataFrame
    data = {
        'product_id': ["#1", "#2"],
        'product_main_category': ['Clothing', 'Accessories'],
        'product_sub_category': ['Shirts', None],
        'color': ['0000', None],
        'component': ['Body', 'Main'],
        'composition_dict': [{'cotton': 50.0, 'polyamide': 10.0}, {'wool': 100.0}],
        'remaining_text': ['Some details', ''],
        'weight': [200.0, 0]
    }
    
    df = pd.DataFrame(data)
    
    # Expected output
    expected_output = [
        ProductDetails(
            product_id='#1',
            category=Category(product_main_category='Clothing', product_sub_category='Shirts'),
            color=Color(color='0000'),
            component=Component(
                component_name='Body',
                composition={'cotton': 50.0, 'polyamide': 10.0},
                additional_details='Some details',
                weight=200.0
            )
        ),
        ProductDetails(
            product_id='#2',
            category=Category(product_main_category='Accessories', product_sub_category=None),
            color=Color(color=None),
            component=Component(
                component_name='Main',
                composition={'wool': 100.0},
                additional_details='',
                weight=0
            )
        )
    ]
    
    # Run the function
    result = dataframe_to_pydantic(df)
    assert result == expected_output, f"Test failed! \nExpected:\n{expected_output}\n\nGot:\n{result}"

def test_pydanticlist_to_json():
    """
    Test function for pydanticlist_to_json to ensure it correctly converts a list of Pydantic models into a JSON file.
    """
    # Sample data for the Pydantic models
    sample_data = [
        ProductDetails(
            product_id='#1',
            category=Category(product_main_category='Clothing', product_sub_category='Shirts'),
            color=Color(color='0000'),
            component=Component(
                component_name='Body',
                composition={'cotton': 50.0, 'polyamide': 10.0},
                additional_details='Some details',
                weight=200.0
            )
        ),
        ProductDetails(
            product_id='#2',
            category=Category(product_main_category='Accessories', product_sub_category='Hats'),
            color=Color(color='Blue'),
            component=Component(
                component_name='Main',
                composition={'wool': 100.0},
                additional_details='Other details',
                weight=150.0
            )
        )
    ]

    # Convert the list of Pydantic models to JSON and save to a file
    pydanticlist_to_json(sample_data, "tests/test_output")

    # Load the JSON file and check its content
    with open("tests/test_output.json", "r") as json_file:
        loaded_data = json.load(json_file)

    # Expected JSON data
    expected_data = [
        {
            "product_id": "#1",
            "category": {
                "product_main_category": "Clothing",
                "product_sub_category": "Shirts"
            },
            "color": {
                "color": "0000"
            },
            "component": {
                "component_name": "Body",
                "composition": {
                    "cotton": 50.0,
                    "polyamide": 10.0
                },
                "additional_details": "Some details",
                "weight": 200.0
            }
        },
        {
            "product_id": "#2",
            "category": {
                "product_main_category": "Accessories",
                "product_sub_category": "Hats"
            },
            "color": {
                "color": "Blue"
            },
            "component": {
                "component_name": "Main",
                "composition": {
                    "wool": 100.0
                },
                "additional_details": "Other details",
                "weight": 150.0
            }
        }
    ]

    # Check if the loaded data matches the expected data
    assert loaded_data == expected_data, f"Test failed! \nExpected:\n{expected_data}\n\nGot:\n{loaded_data}"


if __name__ == "__main__":
    pytest.main()
