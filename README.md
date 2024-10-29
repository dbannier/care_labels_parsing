# Care labels parsing project
Repository containing data for care labels and python code for processing it and transforming it into a json structured format, based on Pydantic models.

## Prerequisites

Create a venv or conda environment prior to installing the requirements, running :

`pip install -r requirements.txt`


To run the code with your own file, store the file in the data/raw directory and run :
`export PYTHONPATH="${PYTHONPATH}:src/"`
`python src/process_labels.py data/raw/{file}.csv`

To test with the loaded file, run:
`export PYTHONPATH="${PYTHONPATH}:src/"`
`python src/process_labels.py data/raw/care_labels.csv`


## Repo architecture and files description:

* data
  * raw : contains the raw data file
  * processed : contains the results files
    * products_database.json : the processed file re-structured in json format, based on Pydantic.
    * final_care_label.csv : the processed file in csv format.
    * to_review.xlsx : specific items that may need special attention as they have remaining text after being processed.
* notebook
  * investigation_nb : jupyter notebook with a brief analysis of the care labels data.
  * process_labels_nb : jupyter notebook processing the file step by step for demo purposes.
* src
  * process_labels : python file with all the processing functions.
  * models: python file containing the pydantic models.
* tests : contains unit test functions. It can be tested by installing pytest and running `pytest` from the terminal.
