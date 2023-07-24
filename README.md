# Financial Inclusion Project (Zindi Competition - closed)

<div id="header" align="center">
  <img src= 'https://media.giphy.com/media/3oKIPEqDGUULpEU0aQ/giphy.gif' width=400>
</div>

<br>

## Introduction
We are:
* Ana 
* Christian 
* Klaus
* Shahi
  
And we are Data Practinioner Trainees at Neuefische Hamburg. 
This is our group Project on the Financial Inclusion Data from Zindi.  
<br>

## Topic of this project:

Financial inclusion remains one of the main obstacles to economic and human development in Africa. For example, across Kenya, Rwanda, Tanzania, and Uganda only 9.1 million adults (or 14% of adults) have access to or use a commercial bank account.

Traditionally, access to bank accounts has been regarded as an indicator of financial inclusion. Despite the proliferation of mobile money in Africa, and the growth of innovative fintech solutions, banks still play a pivotal role in facilitating access to financial services. Access to bank accounts enable households to save and make payments while also helping businesses build up their credit-worthiness and improve their access to loans, insurance, and related services. Therefore, access to bank accounts is an essential contributor to long-term economic growth.

__The objective of this project is to create a machine learning model to predict which individuals are most likely to have or use a bank account.__ The models and solutions developed can provide an indication of the state of financial inclusion in Kenya, Rwanda, Tanzania and Uganda, while providing insights into some of the key factors driving individuals’ financial security.

We also built a web app via streamlit.

<br>

## Data Overview

| column | additional information |
|--------|------------------------|
| country | Country interviewee is in |
| year | Year survey was done in  |
| uniqueid | Unique identifier for each interviewee | 
| location_type | Type of location: Rural, Urban |
| cellphone_access | If interviewee has access to a cellphone: Yes, No |
| household_size | Number of people living in one house |
| age_of_respondent | The age of the interviewee |
| gender_of_respondent | Gender of interviewee: Male, Female | 
| relationship_with_head | The interviewee’s relationship with the head of the house:Head of Household, Spouse, Child, Parent, Other relative, Other non-relatives, Dont know |
| marital_status | The martial status of the interviewee: Married/Living together, Divorced/Seperated, Widowed, Single/Never Married, Don’t know |
| education_level | Highest level of education: No formal education, Primary education, Secondary education, Vocational/Specialised training, Tertiary education, Other/Dont know/RTA |
| job_type | Type of job interviewee has: Farming and Fishing, Self employed, Formally employed Government, Formally employed Private, Informally employed, Remittance Dependent, Government Dependent, Other Income, No Income, Dont Know/Refuse to answer |

<br>

## Requirements and Environment

Requirements for notebooks:
- pyenv with Python: 3.11.3

Streamlit requirements:
scikit-learn==1.2.2
streamlit==1.24.1

Environment: 

For installing the virtual environment you can either use the Makefile and run `make setup` or install it manually with the following commands: 

```Bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

To see the web app you have to first install streamlit and create a virtual environment. Then copy the files from the streamlit folder of this repo into your local streamlit folder. 
Run the python file to run the web app on localhost. 

## Usage

In order to train the model and store test data in the data folder and the model in models run:

```bash
#activate env
source .venv/bin/activate

python example_files/train.py  
```

In order to test that predict works on a test set you created run:

```bash
python example_files/predict.py models/linear_regression_model.sav data/X_test.csv data/y_test.csv
```

## Limitations

Development libraries are part of the production environment, normally these would be separate as the production code should be as slim as possible.

