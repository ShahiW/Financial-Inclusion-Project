# Import modules ---------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error
# -------------------------------------------------------------------
# Dataframes used in this script: 
# data: dataframe with predicted values from Zindi
# df_new:dataframe we generate by filling and saving the survey
# df_ba_yes: dataframe of the data with only the people with a bank account (for plotting purposes in test phase)
# df_knn: new dataframe with predicted value for bank account (with knn model)
# ------------------------------------------------------------------------------------------------------------------
# makes layout wide: (has to be executed first thing after imports)
# st.set_page_config(layout="wide")

# Give the app a name
st.title('Financial inclusion in Africa')

# Fetch data
DATA = 'final_df.csv'     

# Fetch images
image1 = Image.open('055-tanzania_generated.jpg')
image2 = Image.open('vecteezy_kenya-flag-vector_.jpg')
image3 = Image.open('vecteezy_uganda-flag-vector_.jpg')
image4 = Image.open('107-rwanda_generated.jpg') 

# displays list of images
st.image([image1, image2, image3, image4], width=175)


# Cache data to not always load it anew
@st.cache_data
def load_data(nrows):
    # remove thousend seperator comma for year:
    # transform values in column year to string and then replace ',' with '':
    data = pd.read_csv(DATA, nrows=nrows, dtype={'year': str})
    data['year'] = data['year'].str.replace(',', '')
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

# load 34000 rows of data
data = load_data(34000)


# Just for test plotting
# dataframe with only the people with a bank account
df_ba_yes = data.loc[data['bank_account'] == 'Yes']


with st.container():
    # Write the Output
    st.write('Data from 2016-2018 (Tanzania, Kenya, Uganda and Rwanda)')
    st.write(data)

    # Write title
    st.title('Likelihood of having a bank account')
    st.subheader('''By answering the following questions we can predict how likely it is that you (will) have a bank account''')

    # start of survey
    st.write('#### ID')
    id = st.checkbox('Please check the box to start the survey.')
    if id:
        id = 3002
        id += 1
        st.write('Your uniqueid is', id)

    st.write('#### Country')
    country = st.radio(label='##### In which country do you live?', options=['Rwanda', 'Uganda', 'Tanzania', 'Kenya'])

    st.write('#### Location')
    location = st.radio(label='##### Where do you live?', options=['Rural', 'Urban'])

    st.write('#### Gender')
    gender = st.radio(label='##### What is your biological gender?', options=['Male', 'Female'])

    st.write('#### Age')
    age = st.slider('##### How old are you?', 0, 100, 15)
    st.write("I'm ", age, 'years old')

    st.write('#### Relationship within the family')
    relationship = st.radio(label='##### What is your relationship to the head of household?', options=['Head of household', 'Spouse', 'Child', 'Other relative'])

    st.write('#### Household size') 
    household = st.slider('##### How big is your household?', 0, 25, 1)
    st.write( household ,'person/people')

    st.write('#### Marital status')
    marital_status = st.radio(label='##### What is your marital status?', options=['Married/Living together',
                                                                                    'Single/Never married',
                                                                                    'Widowed',
                                                                                    'Divorced/Seperated', 
                                                                                    "Don't know"])

    st.write('#### Education')
    education = st.radio(label='##### What is your level of education?', options=['Primary', 'Secondary', 'Tertiary', 
                                                                            'Vocational/Specialized training', 'Other/Dont know/RTA', 'No formal education'])

    st.write('#### Job type')
    job = st.radio(label='##### What is your job type?', options=['Self employed', 
                                                            'Government dependent',
                                                            'Formally employed private',
                                                            'Informally employed',
                                                            'Formally employed government',
                                                            'Farming and fishing',
                                                            'Remittance dependent',
                                                            'Other income',
                                                            "Don't know/Refuse to answer",
                                                            'No income']
                    )

    st.write('#### Cellphone')
    cellphone = st.radio(label='##### Do you have a cellphone or cellphone access?', options=['Yes', 'No'])


    # save survey
    saved = st.button('Predict')

    # save values in dataframe
    df_new = pd.DataFrame()

    # check if id box was hit
    if saved:
        if id <= 3002:
            st.write('Please check the ID box to submit your survey.')
        #else:
            #st.write('You successfully submitted your survey!')


    # make dataframe of the saved values
    if saved:
        df_new = pd.DataFrame(
            {
                'country': [country],
                'year': ['2018'],
                'uniqueid': [id],
                'bank_account': [None],
                'location_type': [location],
                'cellphone_access': [cellphone],
                'household_size': [household],
                'age_of_respondent': [age],
                'gender_of_respondent': [gender],
                'relationship_with_head': [relationship],
                'marital_status': [marital_status],
                'education_level': [education],
                'job_type': [job]
            }
        ) 


        # modeling
        def predict(df_new):
        
            household_bins = pd.cut(df_new['household_size'], [1, 3, 7, 10, 25], labels=['single', 'small', 'average', 'big'])
            household_bins.name = 'household_sizes'

            df_knn_new = df_new.join(household_bins, how='inner')
            df_knn_new = df_knn_new.drop('household_size', axis=1)

            # Make bins for age 
            age_bins = pd.cut(df_new['age_of_respondent'], [0, 16, 45, 75, 100], labels=['child', 'adult', 'elder', 'old'])
            age_bins.name = 'ages'

            df_knn_new = df_new.join(age_bins, how='inner')
            df_knn_new = df_knn_new.drop('age_of_respondent', axis=1)

            df_knn_new = df_knn_new.drop('uniqueid', axis=1)

            # make dummie variables
            df_knn_new = pd.get_dummies(df_knn_new, drop_first=False)
            
            cols = ['year', 'household_size', 'country_Rwanda', 'country_Tanzania',
                    'country_Uganda', 'bank_account_Yes', 'location_type_Urban',
                    'cellphone_access_Yes', 'gender_of_respondent_Male',
                    'relationship_with_head_Head of Household',
                    'relationship_with_head_Other non-relatives',
                    'relationship_with_head_Other relative',
                    'relationship_with_head_Parent', 'relationship_with_head_Spouse',
                    'marital_status_Dont know', 'marital_status_Married/Living together',
                    'marital_status_Single/Never Married', 'marital_status_Widowed',
                    'education_level_Other/Dont know/RTA',
                    'education_level_Primary education',
                    'education_level_Secondary education',
                    'education_level_Tertiary education',
                    'education_level_Vocational/Specialised training',
                    'job_type_Farming and Fishing', 'job_type_Formally employed Government',
                    'job_type_Formally employed Private', 'job_type_Government Dependent',
                    'job_type_Informally employed', 'job_type_No Income',
                    'job_type_Other Income', 'job_type_Remittance Dependent',
                    'job_type_Self employed', 'ages_adult', 'ages_elder', 'ages_old'
            ]

            import re
            for pos, col in enumerate(cols):
                if re.search("[a-z]_[A-Z]", col) and not col.startswith("bank_account"):
                    if col not in df_knn_new.columns.to_list():
                        df_knn_new.insert(pos, col, [False])
            df_knn_new.insert(0, "year", ["2018"])
            
            df_knn_new = pd.DataFrame(columns=cols, data=df_knn_new)
            df_knn_new = df_knn_new.drop(["bank_account_Yes"], axis=1)

            #st.write('this is the dataframe i have now')
            #st.dataframe(df_knn_new, use_container_width=True)
            
            # Defining X to make prediction for bank account
            features = df_knn_new.columns.tolist()
            X = df_knn_new[features]

            # Predict
            import pickle
            with open('knn.pickle', 'rb') as f:
                knn = pickle.load(f)
            y_pred_final = knn.predict(X)
            # make Series to create column afterwords
            series_knn = pd.Series(y_pred_final).astype(int)  # has to be int, otherwise can't make series
            series_knn = series_knn.astype(bool)  # change back to bool
            #series_knn.name = 'bank_account'

            # concat new columns to new dataframe of saves survey
            #df_knn_new = pd.concat([df_new, series_knn], axis=1)

            # if you want to merge the new dataframe to the old one from zindi:
            # final_df = data.merge(df_new, how='outer')

            # change bool value into Yes or No to fit the values of bank account
            # check first if value is bool
            def bool2yes(boolean):
                if isinstance(boolean, bool):
                    if boolean == True:
                        return "Yes"
                    else:
                        return "No"
                else:
                    return boolean
                
            df_knn_new = df_knn_new.applymap(bool2yes)
            #st.write('prediction')
            #st.success(series_knn)
            if series_knn[0]:
                st.success('Predicted : Yes')
            else:
                st.success('Your result is: No')
            #st.success(f'Your result is: {series_knn[0]}')
        
        predict(df_new)

    # check if id box was hit
    if saved:
        if id <= 3002:
            st.write('Please check the ID box to submit your survey.')
        #else:
            #st.write('You successfully submitted your survey!')






with st.container():
    # OPTIONS FOR PLOTS 
    st.subheader('Visualization')
        
    # select box to select wich plot to show
    option = st.selectbox('##### Which plot do you want to see?',
        ('None', 
        'Heatmap: Country vs bank account', 
        'Histplot: People with bank account by country', 
        'Histplot: People with bank account by job type',
        'Histplot: People with bank account by education level',
        'Countplot: Education level', 
        'Countplot: Job type')
    )


    # PLOTS

    # make heatmap
    if option == 'Heatmap: Country vs bank account':
        st.markdown('### Heatmap')
        fig_heat = px.density_heatmap(data_frame=data, y='bank_account', x='country')
        st.write(fig_heat)


    # make histplot for option: people with bank account by country
    if option == 'Histplot: People with bank account by country':
        st.markdown('### Histplot')
        plot = sns.histplot(data=data, x='country', stat='percent', hue='bank_account', multiple="dodge", shrink=.8)

        plot.set_xticklabels(
        plot.get_xticklabels(), 
        rotation=45, 
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-large'
        
    )   
        
        st.pyplot(plot.get_figure())


    # make histplot for option: people with bank account by job type
    if option == 'Histplot: People with bank account by job type':
        st.markdown('### Histplot')
        custom_palette = sns.color_palette("Paired")
        plot = sns.histplot(data=df_ba_yes, x="country", stat='density', common_norm=False, hue='job_type', multiple="dodge", shrink=.8, palette=custom_palette);
        sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1))

        st.pyplot(plot.get_figure())


    # make histplot for option: people with bank account by job type
    if option == 'Histplot: People with bank account by education level':
        st.markdown('### Histplot')
        custom_palette = sns.color_palette("Paired")
        plot = sns.histplot(data=df_ba_yes, x="country", stat='density', common_norm=False, hue='education_level', multiple="dodge", shrink=.8, palette=custom_palette);
        sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1))

        st.pyplot(plot.get_figure())


    # make countplot for option: education level
    if option == 'Countplot: Education level':
        st.markdown('### Countplot')
        plot = sns.histplot(data=df_ba_yes, x='education_level', hue='country', palette='Set1', stat='density', common_norm=False, multiple='dodge', shrink=.8) #, hue='bank_account')

        plot.set_xticklabels(
        plot.get_xticklabels(), 
        rotation=45, 
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-large'
        
    )   

        st.pyplot(plot.get_figure())


    # make countplot for option: job type
    if option == 'Countplot: Job type':
        st.markdown('### Countplot')
        plot = sns.countplot(data=df_ba_yes, x='job_type', palette='Set3') #, hue='bank_account')

        plot.set_xticklabels(
        plot.get_xticklabels(), 
        rotation=45, 
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-large'
        
    )   

        st.pyplot(plot.get_figure())




# If you want to know some interesting things about the countries in our dataset...
#st.subheader('Did you know? ...')

#st.subheader('How to pronounce Rwanda:')
#st.audio(data='https://upload.wikimedia.org/wikipedia/commons/3/34/Rwanda_%28rw%29_pronunciation.ogg', format="audio/wav", start_time=0, sample_rate=None)

#st.subheader('Greet someone in Kenya:')
#st.write('Hujambo! or more collegial: Jambo! This means Hello! And if someone greets you: Sijambo! That means I am well!')