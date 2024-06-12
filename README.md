# Mushroom Classification
> TDS3301 Data Mining Project || Year 3 Trimester 1

**Group Members**
1. Aina Eirena binti Irwanefandy - 1181203546 - Linear Discriminant Analysis
2. Anis Hazirah binti Mohamad Sabry - 1211300373 - Random Forest Classifier
3. Nuha Awadah binti Mohd Yusof - 1211303209 - Support Vector Machine
4. Raymond Sianjaya - 1181202359 - Gaussian and Categorical Naive Bayes

# Deployment
This project was deployed onto Streamlit under the following url: https://mushroomclassification.streamlit.app/

# Abstract
Mushrooms are a delicate cuisine enjoyed by many. From the lux high class restaurants to far beyond the depths of the forest, it can be found and enjoyed with much delight. But not all mushrooms are created equal. In the midst of its enticing flavours is an enigma of features curating its salivating looks. Experts from Audobon Society Field Guide have collected samples of mushrooms from the Agaricus and Lepiota family, which are now to be used as our dataset to find where the fanciest-looking mushrooms originate.

This project is driven by the overarching goal of analysing the attributes of mushrooms within a specific habitat. The primary objective is to systematically categorise these characteristics and discern the most frequently encountered features along with their corresponding habitats. Through this analysis, we aim to uncover discernible patterns and correlations between the properties of mushrooms and their ecological environments.

Further, the project also ascertained the most effective technique which is Support Vector Machine for predicting the habitat of a given mushroom. Six distinct data mining techniques—Logistic Regression, Linear Discriminant Analysis, Random Forest, Support Vector Machine (SVM), and Naive Bayes—have been applied to the dataset. This study contributes to the existing literature by addressing the gap in understanding how certain mushroom characteristics can predict their habitats. The findings serve as a foundational exploration, paving the way for future research to delve deeper into this intriguing intersection of mushroom properties and ecological environments.

# Introduction
Mushrooms are fascinating and intriguing organisms that have been studied for centuries due to their unique characteristic and diverse habitats. The Mushroom dataset used in this project was collected based on a dataset donated in 1987 and is available on the UCI Machine Learning Repository. This dataset contains information about various mushroom species, including their habitats, cap features, stalk features, smell and print colours.

This project aims to analyse the attributes of mushrooms within a specific habitat to discern unique patterns and correlations between the properties of mushrooms and their ecological environments. By predicting the habitat of a given mushroom, this project can contribute to the existing knowledge on mushroom classification and habitat prediction. In particular, we will apply multiple classification methods to the dataset to find the most effective algorithm and identify which mushroom types correlate with specific habitats.

# Motivation
The project’s primary motivation is to analyse the characteristics of mushrooms found within a habitat. An example would be finding the most frequent cap features within an urban biome. The main objective is to categorise the characteristics found, and find the most commonly occurring features and the habitats they most often appear in. Through this, we will find proof of commonality patterns and correlations between a mushrooms’ property and their current ecological environment.

# Objective
1.	Classify mushroom features according to the most common occurrence within a habitat
2.	Predict the likely characteristics of a mushroom within a habitat

# Dataset
The chosen dataset for our task, titled ‘Mushrooms’, was donated to the UCI’s Machine Learning Repository back in 1987, making the dataset 37 years old. Despite its age, it’s still a dataset that provides great characteristics and features that continues to thrive as a recurrent dataset used by data scientists. The data contains 8124 instances and 22 features comprising descriptions of samples collected from 23 species of gilled mushrooms in the Agaricus and Lepiota family. The samples are labelled as definitely edible, definitely poisonous, of unknown edibility, or not recommended. The dataset originated from a guidebook by the Audubon Society Field Guide, and in it states that there is no clear-cut rule for determining the edibility of a mushroom, as they come in a myriad of features that do not distinctly reflect its edibility value.

The dataset belongs to the domain of Biology and is specifically associated with the subject area of mycology, focusing on mushrooms. The task mostly associated with this dataset is classification. The dataset is sourced from the UCI Machine Learning Repository, a well-known repository for thousands of machine learning and data mining datasets. This dataset is designed where the goal is to classify mushrooms based on various features therefore suitable to be chosen for this classification project.


# Methodology
In this study, we present a framework for analysing the characteristics of mushrooms within specific habitats. The overall pipeline encompasses several key stages, beginning with the collection of data from the Mushroom Dataset available at the UCI Machine Learning Repository.

Following the dataset's acquisition, we conduct an exploratory data analysis (EDA) and implement feature selection using chi-square contingency analysis to enhance the dataset's relevance. Subsequently, the data undergoes preprocessing, including necessary transformations to prepare it for the application of various data mining techniques. This process includes transforming the encoded values to a more comprehensible format and balancing the dataset using SMOTEN.

Six prominent techniques—Logistic Regression, Linear Discriminant Analysis, Random Forest, Support Vector Machine (SVM), Categorical Naive Bayes, and Gaussian Naive Bayes—are employed to predict the habitat of mushrooms based on their distinct features therefore the data preparation section are tailored to accommodate the distinct requirements of each of the six prominent techniques. Evaluation metrics such as comparison  of accuracy scores and using Area Under The Curve (AUC) and Receiver Operating Characteristics (ROC) curve are used to assess the performance of each technique, enabling a comprehensive comparison.

The obtained results and ensuing discussions provide valuable insights into the relationships between mushroom properties and their ecological environments, contributing to the broader understanding of this intriguing domain.

**Steps**
1. Exploratory data analysis
2. Data preprocesssing
> 1. Feature selection
> 2. SMOTEN
3. Data mining models
> 1. Logistic regresion
> 2. Linear Discriminant analysis
> 3. Random Forest Classifier
> 4. Support Vector Machine
> 5. Gaussian Naive Bayes
> 6. Categorical Naive Bayes
4. Evaluation of Models
