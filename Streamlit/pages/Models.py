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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC # "Support vector classifier"
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB

def get_data():
    # GETTING DATASET
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

# PAGE START --------------------------------
st.title("**Classification Techniques for Mushroom Dataset 🍄**")

st.sidebar.write('Algorithms used:')
algolist = ['Logistic Regression', 'Linear Discriminant Analysis (LDA)'
            'Random Forest Classification',
            'Support Vector Machine (SVM)',
            'Gaussian Naive Bayes','Categorical Naive Bayes']
for name in algolist:
    st.sidebar.write(name)

tab1, tab2, tab3 = st.tabs(["Pre-processing", "Models", "Evaluation with ROC"])

with tab1:
    st.title('Pre-processing')
    df = get_data()
    
    # SMOTEN
    target = 'habitat'
    X = df.drop(columns=target)
    y = df['habitat']
    
    with st.expander('Balancing Using SMOTEN'):
        st.header('Before SMOTEN')
        fig = px.bar(df, x=target, title=f'Countplot of {target} Distribution', barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        st.header('After SMOTEN')
        sampler = SMOTEN(random_state=0)
        X_res, y_res = sampler.fit_resample(X, y)
        fig = px.bar(x=y_res, title=f'Countplot of {target} Distribution', barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=1)

    st.write('Training set: ', y_train.shape)
    st.write('Validation set: ', y_val.shape)
    st.write('Testing set: ', y_test.shape)
    
    results = []

    with st.expander('Chi-square analysis'):
        for col in X.columns:
            ct_table_ind = pd.crosstab(X[col], y)

            # compute the chi2 stat and get the value
            c_stat, p, dof, expected = chi2_contingency(ct_table_ind)

            # Interpret p-value
            alpha = 0.05
            result = 'Dependent (reject H0)' if p <= alpha else 'Independent (H0 holds true)'

            # Append results to the list
            results.append([col, c_stat, p, result])

        # Display the DataFrame
        chi_df = pd.DataFrame(results, columns=['Variable', 'Chi2 Statistic', 'P-value', 'Result'])
        st.dataframe(chi_df)

# ENCODING ----------------------------------------------------------------
onehotEncoder = OneHotEncoder()

# for x vals
X_train_onehot = onehotEncoder.fit_transform(X_train)
X_val_onehot = onehotEncoder.transform(X_val)
X_test_onehot = onehotEncoder.transform(X_test)

X_cols_onehot = onehotEncoder.get_feature_names_out()
    
le = LabelEncoder()

# for x vals
# save row numbers
X_train_rows = len(X_train)
X_val_rows = len(X_val)

X_train_val_test= pd.concat(objs=[X_train, X_val, X_test], axis=0) #concatenate the training, validation, and testing attributes

X_train_val_test_le = X_train_val_test.apply(le.fit_transform) #encode categorical attributes to numerical format

#split the training and testing attributes back into individual variables
X_train_le = X_train_val_test_le[:X_train_rows]
X_val_le = X_train_val_test_le[X_train_rows:X_train_rows + X_val_rows]
X_test_le = X_train_val_test_le[X_train_rows + X_val_rows:]


# ROC FUNCS
def print_ROC_scores(y_test, y_prob):
    macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro")
    weighted_roc_auc_ovo = roc_auc_score(
        y_test, y_prob, multi_class="ovo", average="weighted"
    )
    macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    weighted_roc_auc_ovr = roc_auc_score(
        y_test, y_prob, multi_class="ovr", average="weighted"
    )
    st.write(
        "One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
        "(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo)
    )

    st.write()

    st.write(
        "One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
        "(weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr)
    )

target_names = sorted(set(y_test))
label_binarizer = LabelBinarizer().fit(y_train)

def plot_ROC_curve(y_test, y_score, algotitle):
    n_classes = len(np.unique(y_test))
    y_bin = label_binarizer.fit_transform(y_test)

    # Initialize variables for ROC AUC
    roc_auc = dict()

    # Compute ROC curve and ROC area for each class
    fpr_list, tpr_list = [], []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr, tpr)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    # Compute micro-average ROC curve and ROC area
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr_micro, tpr_micro)

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate(fpr_list))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_list[i], tpr_list[i])

    mean_tpr /= n_classes

    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    # Plot ROC curves
    fig = plt.figure(figsize=(8, 6))

    # Plot each class curve
    for i in range(n_classes):
        plt.plot(fpr_list[i], tpr_list[i], label=f'Class {target_names[i]} (AUC = {roc_auc[i]:.2f})')

    # Plot micro and macro average curves
    plt.plot(fpr_micro, tpr_micro, label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(all_fpr, mean_tpr, label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})',
             color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve for ' + algotitle)
    plt.legend(loc='best')
    st.pyplot(fig, use_container_width=True)


accuracy_results = []    

# DATA MINING MODELS ----------------------------------------------------------------
with tab2:
    st.title('Models')
    st.write('Here\'s a look at all the classification techniques we had used in this project!')
    st.write('_Due to the amount of computations going on in the background, Streamlit may take a while to load. Your patience is much appreciated! ⭐_')
    
    # LOGISTIC REGRESSION --------------------------------------------------------------------------------------------------------------------------------
    st.header('Logistic Regression')
    with st.expander('Logistic Regression'):
        st.header('Training and Validation')
        logistic_model = LogisticRegression(max_iter=1000)
        logistic_model.fit(X_train_onehot, y_train)
        y_val_pred = logistic_model.predict(X_val_onehot)

        logistic_accuracy = accuracy_score(y_val, y_val_pred)
        st.write("Validation accuracy score (logistic regression): {:.3f}".format(logistic_accuracy))
        st.write()
        logistic_class_report = classification_report(y_val, y_val_pred)
        st.write("Validation classification report (logistic regression):\n{}".format(logistic_class_report))

        # Create a heatmap for the confusion matrix
        conf_matrix = confusion_matrix(y_val, y_val_pred)

        # Plot confusion matrix heatmap
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=range(6), yticklabels=range(6))
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        st.pyplot(fig, use_container_width=True)
        
        st.header('Training')
        y_test_pred = logistic_model.predict(X_test_onehot)

        # Compute accuracy score
        logistic_test_accuracy = accuracy_score(y_test, y_test_pred)
        st.write("Test accuracy score (logistic regression): {:.3f}".format(logistic_test_accuracy))

        # Compute classification report
        logistic_test_class_report = classification_report(y_test, y_test_pred)
        st.write("Test classification report (logistic regression):\n{}".format(logistic_test_class_report))

        # Create a heatmap for the confusion matrix
        conf_matrix = confusion_matrix(y_test, y_test_pred)

        # Plot confusion matrix heatmap
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=range(6), yticklabels=range(6))
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        st.pyplot(fig, use_container_width=True)

        accuracy_results.append(['Logistic Regression', logistic_accuracy, logistic_test_accuracy])
        y_proba_log = logistic_model.predict_proba(X_test_onehot)
    
    # LDA --------------------------------------------------------------------------------------------------------------------------------
    st.header('Linear Discriminant Analysis')
    with st.expander('Linear Discriminant Analysis'):
        # Apply Linear Discriminant Analysis (LDA)
        lda = LDA()

        X_train_lda = X_train_onehot.copy()
        X_test_lda = X_test_onehot.copy()
        X_val_lda = X_val_onehot.copy()

        X_train_lda = lda.fit_transform(X_train_lda.toarray(), y_train)
        X_test_lda = lda.transform(X_test_lda.toarray())
        X_val_lda = lda.transform(X_val_lda.toarray())
        
        st.header('Training and Validation')
        # Train a logistic regression model on the LDA-transformed data
        logistic_model = LogisticRegression(max_iter=1000)
        logistic_model.fit(X_train_lda, y_train)

        y_val_pred = logistic_model.predict(X_val_lda)

        # Evaluate the model
        st.write("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
        st.write("Validation Classification Report:\n", classification_report(y_val, y_val_pred))
        
        conf_matrix = confusion_matrix(y_val, y_val_pred)

        # Plot confusion matrix heatmap
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=range(6), yticklabels=range(6))
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        st.pyplot(fig, use_container_width=True)
        
        st.header('Testing')
        y_pred = logistic_model.predict(X_test_lda)

        st.write("Testing Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Testing Classification Report:\n", classification_report(y_test, y_pred))

        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d', xticklabels=range(6), yticklabels=range(6))
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        st.pyplot(fig, use_container_width=True)
        
        accuracy_results.append(['LDA', accuracy_score(y_val, y_val_pred), accuracy_score(y_test, y_pred)])
        y_proba_lda = logistic_model.predict_proba(X_test_lda)
            
    # RANDOM FOREST ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    st.header('Random Forest Classification')
    with st.expander('Random Forest'):
        st.header('Finding Best N_Estimators')
        # finding best n_estimators
        scores = {}

        for i in range(1, 50):
            model = RandomForestClassifier(n_estimators=i, random_state=0)
            model = model.fit(X_train_onehot, y_train)

            y_pred = model.predict(X_val_onehot)

            accuracy = accuracy_score(y_val, y_pred)

            scores[i] = accuracy

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        n_vals = [item[0] for item in sorted_scores]
        acc_vals = [item[1] for item in sorted_scores]

        # get the best n_estimator
        best_n_estimator, best_acccuray = sorted_scores[0]

        fig, ax = plt.subplots()
        sns.lineplot(x=n_vals, y=acc_vals, ax=ax)
        ax.set_title('Accuracy vs. n_estimators')
        ax.set_xlabel('n_estimators')
        ax.set_ylabel('Accuracy')

        # Use Streamlit to display the plot
        st.pyplot(fig, use_container_width=True)
        st.write('Top scores and n value', sorted_scores[:10]) # displaying top 10

        
        st.header('Training and Validation')
        clf_RF = RandomForestClassifier(n_estimators=best_n_estimator, random_state=0)
        clf_RF = clf_RF.fit(X_train_onehot, y_train)

        y_val_pred = clf_RF.predict(X_val_onehot)

        st.write("Accuracy score:", accuracy_score(y_val, y_val_pred))

        # Display classification report
        st.write("Classification Report:")
        st.write(classification_report(y_val, y_val_pred))

        # Display confusion matrix
        st.write("Confusion Matrix:")
        conf_matrix = confusion_matrix(y_val, y_val_pred)
        st.write(conf_matrix)

        class_labels = sorted(set(y_val))

        # "d" = decimal integers. formats to decimal integers
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        st.pyplot(fig, use_container_width=True)        
        
        
        st.header('Testing')
        y_pred = clf_RF.predict(X_test_onehot)

        st.write("Accuracy score:", accuracy_score(y_test, y_pred))

        # Display classification report
        st.write("Classification Report:")
        st.write(classification_report(y_test, y_pred))

        # Display confusion matrix
        st.write("Confusion Matrix:")
        conf_matrix = confusion_matrix(y_test, y_pred)
        st.write(conf_matrix)

        # Get the unique class labels
        class_labels = sorted(set(y_test))

        # Plot the heatmap with correct tick labels
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        st.pyplot(fig, use_container_width=True)        

        accuracy_results.append(['Random Forest Classification', accuracy_score(y_val, y_val_pred), accuracy_score(y_test, y_pred)])
        
        st.header('Feature Importance')
        features_data = []

        for i, y_output in enumerate(y_test.unique()):
            # in a multi-class problem, this gets the index of the feature importance per class output
            importance_scores = clf_RF.feature_importances_[i * len(X_train.columns):(i + 1) * len(X_train.columns)]
            feature_importance_dict = dict(zip(X_train.columns, importance_scores))

            # Sort features by importance for the current output
            sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

            # Add data to the list
            for rank, (feature, importance) in enumerate(sorted_features):
                features_data.append({
                    'Habitat': y_output,
                    'Feature': feature,
                    'ImportanceScore': importance,
                })

        # Create DataFrame from the list of dictionaries
        features_df = pd.DataFrame(features_data)

        fig = plt.figure(figsize=(10,10))
        sns.barplot(data=features_df, x='Feature', y='ImportanceScore', hue='Habitat')
        plt.xticks(rotation=90)
        st.pyplot(fig, use_container_width=True)
        
        agg_df = features_df.groupby('Habitat', group_keys=False).apply(lambda x: x.nlargest(10, 'ImportanceScore'))
        st.dataframe(agg_df)
        
        y_proba_rf = clf_RF.predict_proba(X_test_onehot)


    st.header('Support Vector Machine (SVM)')
    with st.expander('SVM - Kernels'):
        # different kernels available ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
        class_labels = ['grasses', 'leaves', 'meadows', 'paths', 'urban', 'woods']

        # loop to test all types of kernels
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        kernels_valaccuracyscore = []
        kernels_accuracyscore = []

        for kernel in kernels:
            st.header(kernel)
            st.write(kernel.upper())

            svc = svm.SVC(kernel=kernel, gamma='auto').fit(X_train_onehot, y_train)

            # Evaluate the model on the validation set
            val_accuracy = svc.score(X_val_onehot.toarray(), y_val)
            st.write(f"Validation Accuracy for {kernel} : {val_accuracy} ")
            # Append the val accuracy score to the list
            kernels_valaccuracyscore.append(val_accuracy)

            # Predict on the test set
            y_pred = svc.predict(X_test_onehot)
            kernel_predaccuracyscore = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy score for {kernel}:", kernel_predaccuracyscore)
            kernels_accuracyscore.append(kernel_predaccuracyscore)

            st.write(" ")

            # Display classification report
            st.write("Classification Report:")
            st.write(classification_report(y_test, y_pred))

            # Display confusion matrix
            st.write("Confusion Matrix:")
            conf_matrix = confusion_matrix(y_test, y_pred)
            st.write(conf_matrix)

            # Plot the heatmap with correct tick labels
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, ax=ax)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title(f'Confusion Matrix for {kernel}')
            st.plotly_chart(fig, use_container_width=True)

        kernels_predval_accuracyscore = pd.DataFrame(list(zip(kernels, kernels_valaccuracyscore, kernels_accuracyscore )), columns=['Kernel', 'Validation Accuracy Score', 'Prediction Accuracy Score'])
        st.dataframe(kernels_predval_accuracyscore)
        # Melt the DataFrame for visualization
        kernels_predval_accuracyscore_melted = pd.melt(kernels_predval_accuracyscore, id_vars='Kernel', var_name='Score Type', value_name='Accuracy Score')

        fig = plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Kernel', y='Accuracy Score', hue='Score Type', data=kernels_predval_accuracyscore_melted)

        plt.xlabel('Kernel', fontsize=12)
        plt.ylabel('Accuracy Score', fontsize=12)
        plt.title('Validation and Prediction Accuracy Scores for Different Kernels', fontsize=14)

        # Displaying the accuracy scores on top of the bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)

        # Move the legend outside the plot
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Show the plot

        plt.ylim(0.85, 0.9)  # Set y-axis limit for better visualization
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander('SVM - Gamma'):
        st.write('In Support Vector Machines (SVM), the gamma parameter is a hyperparameter that defines the influence of a single training example. It affects the shape of the decision boundary and, consequently, the flexibility of the model')
        
        # initialize new lists for gammas accuracy result
        # all
        kernel = []
        kernel_gammas = []
        kernel_gammas_valacc = []
        kernel_gammas_testacc = []

        # displaying only the kernel purposes
        rbf_gamma = []
        rbf_gamma_valacc = []
        rbf_gamma_testacc = []

        poly_gamma = []
        poly_gamma_valacc = []
        poly_gamma_testacc = []

        sigmoid_gamma = []
        sigmoid_gamma_valacc = []
        sigmoid_gamma_testacc = []
        
        st.header('RBF Kernel / Gaussian Kernel')
        st.write('The gamma parameter defines how far the influence of a single training example reaches. Low values mean a far reach, and high values mean a close reach. In other words, a small gamma will result in a more smooth decision boundary, while a large gamma will make the decision boundary more dependent on individual data points.')
        
        gammas = [0.1, 1, 10, 50, 100]

        for gamma in gammas:

            # Append in list to display
            kernel.append('rbf')
            kernel_gammas.append(str(gamma))
            rbf_gamma.append(str(gamma))

            # Fit in SVC
            svc = svm.SVC(kernel='rbf', gamma=gamma).fit(X_train_onehot, y_train)

            # Validation Accuraccy
            val_accuracy = svc.score(X_val_onehot.toarray(), y_val)
            st.write("Validation Accuracy for RBF Kernel of " + str(gamma) + " :" , val_accuracy)
            kernel_gammas_valacc.append(val_accuracy)
            rbf_gamma_valacc.append(val_accuracy)

            # Test Accuracy
            y_pred = svc.predict(X_test_onehot)
            pred_accuracy = accuracy_score(y_test, y_pred)
            st.write("Prediction Accuracy for RBF Kernel of " + str(gamma) + " :" , pred_accuracy)
            kernel_gammas_testacc.append(pred_accuracy)
            rbf_gamma_testacc.append(pred_accuracy)
            
            st.write()
        # Plotting
        fig = plt.figure(figsize=(10, 6))

        # Plot validation accuracy
        plt.plot(gammas, rbf_gamma_valacc , marker='o', label='Validation Accuracy')

        # Plot test accuracy
        plt.plot(gammas, rbf_gamma_testacc, marker='s', label='Test Accuracy')

        # Adding labels and title
        plt.title('Accuracy Scores for RBF Kernel with Different Gamma Values')
        plt.xlabel('Gamma')
        plt.ylabel('Accuracy')
        plt.xscale('log')  # Set x-axis to logarithmic scale for better visualization
        plt.xticks(gammas, gammas)  # Set gamma values as ticks on x-axis
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
    with st.expander('SVM - Polynomial Kernel'):
        st.write('The gamma parameter influences the shape of the decision boundary. It is related to the inverse of the radius of influence of samples selected by the model. Larger gamma values lead to more complex decision boundaries.')
        # poly kernel
        gammas = [0.1, 0.2, 0.4] ## it took a while for 10

        for gamma in gammas:

            # Append in list to display
            kernel.append('poly')
            kernel_gammas.append(str(gamma))
            poly_gamma.append(str(gamma))

            # Fit in SVM
            svc = svm.SVC(kernel='poly', gamma=gamma).fit(X_train_onehot, y_train)

            # Validation Accuraccy
            val_accuracy = svc.score(X_val_onehot.toarray(), y_val)
            st.write("Validation Accuracy for Poly Kernel of " + str(gamma) + " :" , val_accuracy)
            kernel_gammas_valacc.append(val_accuracy)
            poly_gamma_valacc.append(val_accuracy)

            # Test Accuracy
            y_pred = svc.predict(X_test_onehot)
            pred_accuracy = accuracy_score(y_test, y_pred)
            st.write("Prediction Accuracy for Poly Kernel of " + str(gamma) + " :" , pred_accuracy)
            kernel_gammas_testacc.append(pred_accuracy)
            poly_gamma_testacc.append(pred_accuracy)
            st.write()
        # Plotting
        fig = plt.figure(figsize=(10, 6))

        # Plot validation accuracy
        plt.plot(gammas, poly_gamma_valacc , marker='o', label='Validation Accuracy')

        # Plot test accuracy
        plt.plot(gammas, poly_gamma_testacc, marker='s', label='Test Accuracy')

        # Adding labels and title
        plt.title('Accuracy Scores for Poly Kernel with Different Gamma Values')
        plt.xlabel('Gamma')
        plt.ylabel('Accuracy')
        plt.xscale('log')  # Set x-axis to logarithmic scale for better visualization
        plt.xticks(gammas, gammas)  # Set gamma values as ticks on x-axis
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
            
    with st.expander('SVM - Sigmoid Kernel'):
        # sigmoid kernel
        gammas = [0.1, 1, 10, 50, 100]

        for gamma in gammas:

            # Append in list to display
            kernel.append('sigmoid')
            kernel_gammas.append(str(gamma))
            sigmoid_gamma.append(str(gamma))

            # Fit in SVM
            svc = svm.SVC(kernel='sigmoid', gamma=gamma).fit(X_train_onehot, y_train)

            # Validation Accuraccy
            val_accuracy = svc.score(X_val_onehot.toarray(), y_val)
            st.write("Validation Accuracy for Sigmoid Kernel of " + str(gamma) + " :" , val_accuracy)
            kernel_gammas_valacc.append(val_accuracy)
            sigmoid_gamma_valacc.append(val_accuracy)

            # Test Accuracy
            y_pred = svc.predict(X_test_onehot)
            pred_accuracy = accuracy_score(y_test, y_pred)
            st.write("Prediction Accuracy for Sigmoid Kernel of " + str(gamma) + " :" , pred_accuracy)
            kernel_gammas_testacc.append(pred_accuracy)
            sigmoid_gamma_testacc.append(pred_accuracy)
            st.write()
        # Plotting
        fig = plt.figure(figsize=(10, 6))

        # Plot validation accuracy
        plt.plot(gammas, sigmoid_gamma_valacc , marker='o', label='Validation Accuracy')

        # Plot test accuracy
        plt.plot(gammas, sigmoid_gamma_testacc, marker='s', label='Test Accuracy')

        # Adding labels and title
        plt.title('Accuracy Scores for Poly Kernel with Different Gamma Values')
        plt.xlabel('Gamma')
        plt.ylabel('Accuracy')
        plt.xscale('log')  # Set x-axis to logarithmic scale for better visualization
        plt.xticks(gammas, gammas)  # Set gamma values as ticks on x-axis
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
            
    with st.expander('SVM - Comparision of each Kernal and Gamma Values'):
        # Displaying the different kernels and different value for gammas
        gamma_predval_accuracyscore = pd.DataFrame(list(zip(kernel, kernel_gammas, kernel_gammas_valacc, kernel_gammas_testacc )), columns=['Kernel', 'Gamma Value', 'Validation Accuracy Score', 'Prediction Accuracy Score'])
        st.dataframe(gamma_predval_accuracyscore)

        # Find the maximum validation accuracy score and its corresponding row
        max_valacc_row = gamma_predval_accuracyscore.loc[gamma_predval_accuracyscore['Validation Accuracy Score'].idxmax()]
        # Find the maximum prediction accuracy score and its corresponding row
        max_testacc_row = gamma_predval_accuracyscore.loc[gamma_predval_accuracyscore['Prediction Accuracy Score'].idxmax()]

        # Display the rows with maximum scores
        st.write("Maximum Validation Accuracy Score:")
        st.write(max_valacc_row)

        st.write("\nMaximum Prediction Accuracy Score:")
        st.write(max_testacc_row)
        
        # Plotting
        bar_width = 0.35
        index = np.arange(len(kernel_gammas_valacc))

        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot validation accuracy scores
        val_bars = ax.bar(index - bar_width/2, kernel_gammas_valacc, bar_width, label='Validation Accuracy')

        # Plot prediction accuracy scores
        pred_bars = ax.bar(index + bar_width/2, kernel_gammas_testacc, bar_width, label='Prediction Accuracy')

        # Adding labels and title
        ax.set_xlabel('Kernel and Gamma Value')
        ax.set_ylabel('Accuracy Scores')
        ax.set_title('Accuracy Scores for Different Kernels and Gamma Values')
        ax.set_xticks(index)
        ax.set_xticklabels([f'{kernel[i]}\nGamma={kernel_gammas[i]}' for i in range(len(kernel_gammas))])
        ax.legend()

        # Displaying the accuracy scores on top of the bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=7)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # Plotting back the graph
        fig = plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Kernel', y='Accuracy Score', hue='Score Type', data=kernels_predval_accuracyscore_melted)

        plt.xlabel('Kernel', fontsize=12)
        plt.ylabel('Accuracy Score', fontsize=12)
        plt.title('Validation and Prediction Accuracy Scores for Different Kernels', fontsize=14)

        # Displaying the accuracy scores on top of the bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)

        # Move the legend outside the plot
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Show the plot

        plt.ylim(0.85, 0.9)  # Set y-axis limit for better visualization
        st.pyplot(fig, use_container_width=True)
        
        st.dataframe(kernels_predval_accuracyscore)

        # Find the row with maximum validation accuracy score
        max_valacc_row = kernels_predval_accuracyscore.loc[kernels_predval_accuracyscore['Validation Accuracy Score'].idxmax()]

        # Find the row with maximum prediction accuracy score
        max_testacc_row = kernels_predval_accuracyscore.loc[kernels_predval_accuracyscore['Prediction Accuracy Score'].idxmax()]

        # Display the rows with maximum scores
        st.write("Maximum Validation Accuracy Score:")
        st.write(max_valacc_row)

        st.write("\nMaximum Prediction Accuracy Score:")
        st.write(max_testacc_row)
    
    st.header('Naive Bayes')        
    with st.expander('Gaussian Naive Bayes'):
        st.header('Training and Validation')
        Gnb = GaussianNB()
        Gnb.fit(X_train_le, y_train)
        Gnb_pred = Gnb.predict(X_val_le)

        #classification report
        st.write("Classification Report:")
        st.write(classification_report(y_val, Gnb_pred,target_names=class_labels))

        #confusion matrix
        st.write("Confusion Matrix:")
        Gnb_conf_mat = confusion_matrix(y_val, Gnb_pred, normalize='true')
        st.write(Gnb_conf_mat,'\n')

        #heat map
        fig, ax = plt.subplots()
        sns.heatmap(Gnb_conf_mat, annot=True, xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Gaussian Confusion Matrix')
        st.pyplot(fig, use_container_width=True)    
        
        st.header('Testing')
        Gnb_pred = Gnb.predict(X_test_le)

        #classification report
        st.write("Classification Report:")
        st.write(classification_report(y_test, Gnb_pred,target_names=class_labels))

        #confusion matrix
        st.write("Confusion Matrix:")
        Gnb_conf_mat = confusion_matrix(y_test, Gnb_pred, normalize='true')
        st.write(Gnb_conf_mat,'\n')

        #heat map
        fig, ax = plt.subplots()
        sns.heatmap(Gnb_conf_mat, annot=True, xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Gaussian Confusion Matrix')
        st.pyplot(fig, use_container_width=True)
        
        st.write('Gaussian Naive Bayes')
        st.write('Training accuracy: ', Gnb.score(X_val_le, y_val))
        st.write('Validation accuracy: ', Gnb.score(X_test_le, y_test),'\n')  
            
    with st.expander('Categorical Naive Bayes'):
        st.header('Training and Validation')
        Cnb = CategoricalNB()
        Cnb.fit(X_train_le, y_train)
        Cnb_pred = Cnb.predict(X_val_le)

        #classification report
        st.write("Classification Report:")
        st.write(classification_report(y_val, Cnb_pred,target_names=class_labels))

        #confusion matrix
        st.write("Confusion Matrix:")
        Cnb_conf_mat = confusion_matrix(y_val, Cnb_pred, normalize='true')
        st.write(Cnb_conf_mat,'\n')

        #heat map
        fig, ax = plt.subplots()
        sns.heatmap(Cnb_conf_mat, annot=True, xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Categorical Confusion Matrix')
        st.pyplot(fig, use_container_width=True)
        
        st.header('Testing')
        Cnb_pred = Cnb.predict(X_test_le)

        #classification report
        st.write("Classification Report:")
        st.write(classification_report(y_test, Cnb_pred,target_names=class_labels))

        #confusion matrix
        st.write("Confusion Matrix:")
        Cnb_conf_mat = confusion_matrix(y_test, Cnb_pred, normalize='true')
        st.write(Cnb_conf_mat,'\n')

        #heat map
        fig, ax = plt.subplots()
        sns.heatmap(Cnb_conf_mat, annot=True, xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Categorical Confusion Matrix')
        st.pyplot(fig, use_container_width=True)

        st.write('Categorical Naive Bayes')
        st.write('Training accuracy: ', Cnb.score(X_val_le, y_val))
        st.write('Validation accuracy: ', Cnb.score(X_test_le, y_test))

        accuracy_results.append(['Gaussian Naive Bayes',  Gnb.score(X_val_le, y_val), Gnb.score(X_test_le, y_test)])
        accuracy_results.append(['Categorical Naive Bayes',  Cnb.score(X_val_le, y_val), Cnb.score(X_test_le, y_test)])
    
    st.header('Accuracy Comparision')
    results_table = pd.DataFrame(accuracy_results, columns=['Model', 'Validation Accuracy', 'Testing Accuracy'])
    st.dataframe(results_table)
    
    st.subheader('SVM')
    st.dataframe(gamma_predval_accuracyscore)
    st.dataframe(kernels_predval_accuracyscore)

with tab3:
    with st.expander('Logistic Regression'):
        print_ROC_scores(y_test, y_proba_log)
        plot_ROC_curve(y_test, y_proba_log, 'Logistic Regression')

    with st.expander('LDA'):
        print_ROC_scores(y_test, y_proba_lda)
        plot_ROC_curve(y_test, y_proba_lda, 'LDA')
    
    with st.expander('Random Forest'):
        print_ROC_scores(y_test, y_proba_rf)
        plot_ROC_curve(y_test, y_proba_rf, 'Random Forest Classification')
    
    with st.expander('SVM'):
        svm_model = SVC(kernel='rbf', gamma='auto', probability=True)
        svm_model = svm_model.fit(X_train_onehot, y_train)
        # Printing ROC Curves
        y_proba = svm_model.predict_proba(X_test_onehot)
        print_ROC_scores(y_test, y_proba)
        plot_ROC_curve(y_test, y_proba, 'SVM RBF Kernel Model')
        
        y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))

        # Fit SVM model
        svm_model = OneVsRestClassifier(SVC(kernel='rbf', gamma='auto', probability=True)).fit(X_train_onehot, y_train)

        # Generate predicted probabilities
        y_scores = svm_model.predict_proba(X_test_onehot)

        # Calculate ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(np.unique(y_train))):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve for each class
        fig = plt.figure(figsize=(8, 6))
        for i in range(len(np.unique(y_train))):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for SVM (RBF Kernel) - One-vs-Rest')
        plt.legend(loc="lower right")
        st.pyplot(fig, use_container_width=True)
    
    with st.expander('Naive Bayes'):
        y_proba_gnb = Gnb.predict_proba(X_test_le)
        print_ROC_scores(y_test, y_proba_gnb)
        plot_ROC_curve(y_test, y_proba_gnb, 'Gaussian Naive Bayes')
        
        y_proba_cnb = Cnb.predict_proba(X_test_le)
        print_ROC_scores(y_test, y_proba_cnb)
        plot_ROC_curve(y_test, y_proba_cnb, 'Categorical Naive Bayes')