# Care labels parsing project
Repository containing data for care labels and python code for processing it and transforming it into a json structured format.

## Prerequisites

Create a venv or conda environment prior to installing the requirements, running : 

`pip install -r requirements.txt``

## Repo architecture and files description:

* data
  * raw : contans the raw data file 
  * processed : contains the results files 
* notebook 
  * investigation_nb : jupyter notebook with a brief analysis of the care labels data.
  * process_labels_nb : jupyter notebook executing the src functions and displays results step by step. 
* src
  * process_labels : python file with all the processing functions.
  * models: python file containing the pydantic models.
