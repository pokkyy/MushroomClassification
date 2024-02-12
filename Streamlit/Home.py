import streamlit as st
import pandas as pd
import plotly.express as px
from ucimlrepo import fetch_ucirepo

# SIDEBAR ---------------------------------------------------------------------------------------------------

st.sidebar.write('**Mushrooms Classification**')
st.sidebar.write('TDS3301 Data Mining Project Final')

st.title("**Classification Techniques for Mushroom Dataset**")

# abstract
st.write('_Mushrooms are a delicate cuisine enjoyed by many. From the lux high class restaurants to far beyond the depths of the forest, it can be found and enjoyed with much delight. ' +
         'But in the midst of its flavour, is the mechanism to protect itself that evolved as the world turns around it. ' +
         'That is, poison. Experts from Audobon Society Field Guide have collected samples of mushrooms, with the aim to distinguish those that can bless the world with its taste or those that will leave nothing but death in its wake._')
st.write('_This project is driven by the overarching goal of analyzing the attributes of mushrooms within a specific habitat. ' +
         'The primary objective is to systematically categorize these characteristics and discern the most frequently encountered features along with their corresponding habitats. ' +
         'Through this analysis, we aim to uncover discernible patterns and correlations between the properties of mushrooms and their ecological environments._')
    
with st.expander("**Expected Output:**"):
    st.write('- Classify mushroom features according to the most common occurrence within a habitat\n',
             '- Predict the likely characteristics of a mushroom within a habitat\n',
             '- Find the most efficient algorithm that is able to do the above tasks. Predict and categorise the mushroom features to their habitats.')

tab1, tab2, tab3 = st.tabs(["Group Members", "Exploratory Data Analysis", "Findings"])

with tab1:
    st.title('Group 6 Members')
    members = [['AINA EIRENA BINTI IRWANEFANDY', '1181203546', 'TC1L  / TT2L', 'Logistic Regression and Linear Discriminant Analysis (LDA)',],
               ['ANIS HAZIRAH BINTI MOHAMAD SABRY', '1211300373', 'TC2L / TT3L', 'Random Forest Classification'],
               ['NUHA AWADAH BINTI MOHD YUSOF', '1211303209', 'TC2L / TT3L', 'Support Vector Machine (SVM)'],
               ['RAYMOND SIANJAYA', '1181202359', 'TC1L / TT2L', 'Gaussian Naive Bayes and Categorical Naive Bayes']]
    
    group_df = pd.DataFrame(members, columns=['Name', 'Student ID', 'Lecture / Tutorial Section', 'Algorithm'])
    st.table(group_df)

# DF --------------------------------------------------------------------------------------------------
with tab2:
    st.title("Mushrooms Database")
    st.write('The primary motivation of this project is to analyse the characteristics of mushrooms found within a habitat. An example would be finding the most frequent cap features within an urban biome.',
            'The main objective is to categorise the characteristics found, and categorise the most commonly occurring features and  the habitats they most often appear in.',
            'Through this, we will find proof of commonality patterns and correlations between the property of a mushroom and their current ecological environment.')
    
    st.header('A quick glance at the data')
    mushroom = fetch_ucirepo(id=73)
    X = mushroom.data.features
    y = mushroom.data.targets
    raw = pd.concat([X, y], axis=1)

    # extract codebook to mapping dict
    mushroom_map = {'poisonous': {'p':'poisonous', 'e':'edible'}}
    mushroom_vars = pd.DataFrame(mushroom.variables)

    # display(mushroom.variables)

    for index, row in mushroom_vars.iloc[1:].iterrows():
        # split the description items
        var = str(row['name'])
        desc = str(row['description'])
        desc_items = desc.split(',')

        if var not in mushroom_map:
            mushroom_map[var] = {}

        # add dictionary of descriptions
        for item in desc_items:
            value, key = item.split('=')
            mushroom_map[var][key.strip()] = value.strip()
    df_rename = raw.copy()

    for col, mapping in mushroom_map.items():
        if col in df_rename.columns:
            df_rename[col] = df_rename[col].apply(lambda x: mapping.get(x, x))
    df = df_rename.dropna()
    st.dataframe(df)
    
    df.describe()
    
    # EDA
    
    st.header('Exploratory Data Analysis')
    cap_vars = df.columns[:3]
    gill_vars = df.columns[5:9]
    stalk_vars = df.columns[9:15]
    veil_vars = df.columns[15:17]
    ring_vars = df.columns[17:19]
    pop_vars = df.columns[20:-1] # 'pop' = population, and habitat
    other_vars = ['bruises', 'odor', 'spore-print-color']
    
    all_vars = [cap_vars, gill_vars, stalk_vars, veil_vars, ring_vars, pop_vars, other_vars]
    interactive_titles = ['Cap', 'Gill', 'Stalk', 'Veil', 'Ring', 'Population', 'Other']
    
    for title, vars in zip(interactive_titles, all_vars):
        with st.expander(f'Interactive Countplot of Mushroom {title} Variables'):
            for col in vars:
                fig = px.histogram(df, x=col, color='poisonous', title=f'Countplot of {col}', barmode='group')
                st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.write('CONCLUSION')