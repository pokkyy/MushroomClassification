import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from ucimlrepo import fetch_ucirepo

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from imblearn.over_sampling import SMOTEN
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

def get_data():
    mushroom = fetch_ucirepo(id=73)
    X = mushroom.data.features
    y = mushroom.data.targets
    raw = pd.concat([X, y], axis=1)

    # extract codebook to mapping dict
    mushroom_map = {'poisonous': {'p':'poisonous', 'e':'edible'}}
    mushroom_vars = pd.DataFrame(mushroom.variables)

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
    
    return df

df = get_data()

target = ['habitat']
X = df.drop(columns=target)
y = df['habitat']

# Preparing the dataset
sampler = SMOTEN(random_state=0)
X_res, y_res = sampler.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=1)

features = [""] * 22 

def features_input():
    feature_select = [item for item in X.columns if item != 'habitat']
    for i, ft in enumerate(feature_select):
        feature_name = 'Select a ' + str(ft)
        features[i] = st.radio(
            feature_name,
            df[feature_select[i]].unique(),
            horizontal=True,
        )
    return features

models = ['Logistic Regression', 'Linear Discriminant Analysis', 'Random Forest Classifier', 'Support Vector Machine',  'Naive Bayes']
svm_options = ['linear', 'poly', 'rbf', 'sigmoid']
nb_options = ['Gaussian', 'Categorical']

def get_modeling(features, model, model_extra):
    X_test = [features]
    
    # ENCODING ----------------------------------------------------------------
    onehotEncoder = OneHotEncoder()
    X_train_onehot = onehotEncoder.fit_transform(X_train)
    X_val_onehot = onehotEncoder.transform(X_val)
    X_test_onehot = onehotEncoder.transform(X_test)

    le = LabelEncoder()
    X_train_rows = len(X_train)
    X_val_rows = len(X_val)
    X_train_val_test = pd.concat([X_train, X_val, pd.DataFrame(X_test, columns=X_train.columns)], axis=0)
    X_train_val_test_le = X_train_val_test.apply(le.fit_transform) #encode categorical attributes to numerical format
    #split the training and testing attributes back into individual variables
    X_train_le = X_train_val_test_le[:X_train_rows]
    X_val_le = X_train_val_test_le[X_train_rows:X_train_rows + X_val_rows]
    X_test_le = X_train_val_test_le[X_train_rows + X_val_rows:]

    lda = LDA()
    X_train_lda = X_train_onehot.copy()
    X_test_lda = X_test_onehot.copy()
    X_train_lda = lda.fit_transform(X_train_lda.toarray(), y_train)
    X_test_lda = lda.transform(X_test_lda.toarray())
    
    pred = None
    if model_extra:
        if model == models[3]:
            pred = get_SVM(model_extra, X_train_onehot, X_test_onehot)
        elif model == models[4]:
            if model_extra == nb_options[0]:
                pred = get_gnb(X_train_le, X_test_le)
            elif model_extra == nb_options[1]:
                pred = get_cnb(X_train_le, X_test_le)
    else:
        if model == models[0]:
            pred = get_logreg(X_train_onehot, X_test_onehot)
        elif model == models[1]:
            pred = get_lda(X_train_lda, X_test_lda)
        elif model == models[2]:
            pred = get_rf(X_train_onehot, X_val_onehot, X_test_onehot)
    
    return pred

# MODELS ----------------------------------------------------------------
def get_logreg(X_train_onehot, X_test_onehot):
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train_onehot, y_train)

    pred = logistic_model.predict(X_test_onehot)
    return pred

def get_lda(X_train_lda, X_test_lda):
    lda_logistic_model = LogisticRegression(max_iter=1000)
    lda_logistic_model.fit(X_train_lda, y_train)
    
    pred = lda_logistic_model.predict(X_test_lda)
    return pred

def get_rf(X_train_onehot, X_val_onehot, X_test_onehot):
    scores = {}
    for i in range(1, 50):
        model = RandomForestClassifier(n_estimators=i, random_state=0)
        model = model.fit(X_train_onehot, y_train)
        y_pred = model.predict(X_val_onehot)
        accuracy = accuracy_score(y_val, y_pred)
        scores[i] = accuracy
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # get the best n_estimator
    best_n_estimator, best_acccuray = sorted_scores[0]
    
    clf_RF = RandomForestClassifier(n_estimators=best_n_estimator, random_state=0)
    clf_RF = clf_RF.fit(X_train_onehot, y_train)

    pred = clf_RF.predict(X_test_onehot)
    return pred

def get_SVM(kernel, X_train_onehot, X_test_onehot):   
    svc = svm.SVC(kernel=kernel, gamma='auto').fit(X_train_onehot, y_train)
    pred = svc.predict(X_test_onehot)
    
    return pred

def get_gnb(X_train_le, X_test_le):
    Gnb = GaussianNB()
    Gnb.fit(X_train_le, y_train)
    
    pred = Gnb.predict(X_test_le)
    return pred

def get_cnb(X_train_le, X_test_le):
    Cnb = CategoricalNB()
    Cnb.fit(X_train_le, y_train)
    
    pred = Cnb.predict(X_test_le)
    return pred
    
# SIDEBAR ---------------------------------------------------------------------------------------------------

st.sidebar.write('**Mushrooms Classification 🔎**')
st.sidebar.write('TDS3301 Data Mining Project Final')

st.title("**Classification Techniques for Mushroom Dataset 🍄**")

# abstract
st.write('_Mushrooms are a delicate cuisine enjoyed by many. From the lux high class restaurants to far beyond the depths of the forest, it can be found and enjoyed with much delight. ' +
         'But in the midst of its flavour, is the mechanism to protect itself that evolved as the world turns around it. ' +
         'That is, poison. Experts from Audobon Society Field Guide have collected samples of mushrooms, with the aim to distinguish those that can bless the world with its taste or those that will leave nothing but death in its wake._')
st.write('_This project is driven by the overarching goal of analyzing the attributes of mushrooms within a specific habitat. ' +
         'The primary objective is to systematically categorize these characteristics and discern the most frequently encountered features along with their corresponding habitats. ' +
         'Through this analysis, we aim to uncover discernible patterns and correlations between the properties of mushrooms and their ecological environments._')

with st.expander('**Motivation and Objective**'):
    st.write('The primary motivation of this project is to analyse the characteristics of mushrooms found within a habitat. An example would be finding the most frequent cap features within an urban biome.',
        'The main objective is to categorise the characteristics found, and categorise the most commonly occurring features and  the habitats they most often appear in.',
        'Through this, we will find proof of commonality patterns and correlations between the property of a mushroom and their current ecological environment.')
with st.expander("**Expected Output:**"):
    st.write('- Classify mushroom features according to the most common occurrence within a habitat\n',
             '- Predict the likely characteristics of a mushroom within a habitat\n',
             '- Find the most efficient algorithm that is able to do the above tasks. Predict and categorise the mushroom features to their habitats.')

tab1, tab2, tab3, tab4 = st.tabs(["Group Members", "Exploratory Data Analysis", "Findings", "Make Your Own Predictions!"])

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
    st.title("Mushrooms Database 🗂️")
    st.write('The chosen dataset for our task, titled \'Mushrooms\', was donated to the UCI\'s Machine Learning Repository back in 1987, making the dataset 37 years old!',
             'Despite its age, it\'s still a dataset that provides great characteristics and features that continues to thrive as a recurrent dataset used by data scientists.')
    st.write('The data contains 8124 instances and 22 features comprising descriptions of samples collected from 23 species of gilled mushrooms in the Agaricus and Lepiota family.',
             'The samples are labelled as definitely edible, definitely poisonous, of unknown edibility, or not recommended.',
             'The dataset originated from a guidebook by the Audubon Society Field Guide, and in it states that there is no clear-cut rule for determining the edibility of a mushroom, as they come in a myriad of features that do not distinctly reflect its edibility value.')
    st.write('The dataset belongs to the domain of Biology and is specifically associated with the subject area of mycology, focusing on mushrooms.',
             'The task mostly associated with this dataset is classification. The dataset is sourced from the UCI Machine Learning Repository, a well-known repository for thousands of machine learning and data mining datasets.',
             'This dataset is designed where the goal is to classify mushrooms based on various features therefore suitable to be chosen for this classification project.')
    
    st.header('A quick glance at the data')
    st.dataframe(df)
    
    st.header('Description of the Data')
    st.write(df.describe())
    
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
    
    # for title, vars in zip(interactive_titles, all_vars):
    #     with st.expander(f'Interactive Countplot of Mushroom {title} Variables'):
    #         for col in vars:
    #             fig = px.histogram(df, x=col, color='poisonous', title=f'Countplot of {col}', barmode='group')
    #             st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.write('CONCLUSION')

with tab4:
    st.title('Where would I find this mushroom?')
    st.write('For more information on how our models work, see the \'Models\' page!')
    
    features = features_input()
    features
        
    model_select = st.selectbox(
        "Select your choice of model",
        models,
        index=None,
    )
    st.write(f"Selected Model: {model_select}")
        
    if model_select:
        model_extra = ""
        
        if model_select == models[-2]:
            model_extra = st.selectbox(
                "Select your choice of SVM Kernel",
                svm_options,
            )
            st.write(f"SVM Kernel chosen: {model_extra}")
        elif model_select == models[-1]:
            model_extra = st.selectbox(
                "Select your choice of Naive Bayes",
                nb_options,
            )
            st.write(f"Naive Bayes variant chosen: {model_extra}")
        
        pred = get_modeling(features, model_select, model_extra)
        
        if pred:
            st.write('With thoese features, you will most likely find your mushroom in this habitat!')
            st.write(pred)
    else:
        st.write('Please wait for all the features to appear once everything is filled! :)')
