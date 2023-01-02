# library doc string
'''
Predict customer churn project main script file.
Author: Paul Hake
Date: 12/30/22
'''
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_curve

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)


def perform_eda(data):
    '''
    perform eda on df and save figures to images folder
    input:
            data: pandas dataframe
    output:
            None
    '''
    # refactor target variable from categorical to 1 is attrited and 0 if
    # remains
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # check for nulls and store as csv file
    data.isnull().sum().to_csv('./data/nulls_check.csv')

    # check univariate statistics of data and store as csv
    data.describe().to_csv('./data/descriptive_stats.csv')

    # check for class imbalance and save chart image
    plt.figure(figsize=(20, 10))
    data['Churn'].hist()
    plt.savefig('./images/eda/Churn_hist.png')
    plt.close()

    # check Customer_Age distribution and save chart image
    plt.figure(figsize=(20, 10))
    data['Customer_Age'].hist()
    plt.savefig('./images/eda/customer_age_hist.png')
    plt.close()

    # check Marital_Status distribution and save chart image
    plt.figure(figsize=(20, 10))
    data.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/marital_status_hist.png')
    plt.close()

    # check distribution of TOtal_Trans_ct variable and save chart image
    plt.figure(figsize=(20, 10))
    sns.distplot(data['Total_Trans_Ct'], kde=True)
    plt.savefig('./images/eda/Total_Trans_Ct_distribution.png')
    plt.close()

    # check bivariate distribution with heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(data.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/data_correlation_heatmap.png')
    plt.close()

    return data


def encoder_helper(dataframe, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from
    the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that
            could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    encoded_dataframe = dataframe.copy(deep=True)
    for col in category_lst:
        encoded_dataframe[col + '_Churn'] = encoded_dataframe.groupby(col)[
            response].transform('mean')
    return encoded_dataframe


def perform_feature_engineering(dataframe, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that
              could be used for naming variables or index y column]
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # list of categorical columns
    category_lst = ['Gender',
                    'Education_Level',
                    'Marital_Status',
                    'Income_Category',
                    'Card_Category']

    # create new columns for proportion of Churn in each category
    encoded_dataframe = encoder_helper(
        dataframe=dataframe,
        category_lst=category_lst,
        response=response)

    # create target var y from Churn variable
    y = encoded_dataframe['Churn']

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    final_data = pd.DataFrame()
    final_data[keep_cols] = encoded_dataframe[keep_cols]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(final_data, y,
                                                        test_size=0.3,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # random forest (rf) and logistic regression (lr) models
    rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000, n_jobs=-1)

    # grid search parameters for rf model
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    # run grid search and fit for rf model
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # fit lr model
    lrc.fit(X_train, y_train)

    # create train/test results for both lr and rf models
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # save results to csv files
    np.savetxt(
        './data/y_train_preds_rf.csv',
        y_train_preds_rf,
        delimiter='csv')
    np.savetxt('./data/y_test_preds_rf.csv', y_test_preds_rf, delimiter='csv')
    np.savetxt(
        './data/y_train_preds_lr.csv',
        y_train_preds_lr,
        delimiter='csv')
    np.savetxt('./data/y_test_preds_lr.csv', y_test_preds_lr, delimiter='csv')

    # save best models as pickle file
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # classification report
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # Compute and feature importance
    feature_importance_plot(
        model=cv_rfc,
        X_data=X_test,
        output_pth='./images/results/cv_rfc_model_feature_importance.png')

    # roc curve plots for random forest and logistic regression
    roc_plot(cv_rfc.best_estimator_, X_test, y_test)
    roc_plot(lrc, X_test, y_test)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # classification report for random forest models save to image file
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/classification_report_random_forest.png')
    plt.close()

    # classification report for logistic regression model save to image file
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/classification_report_logistic_regression.png')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances and save report as png file
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth)
    plt.close()


def roc_plot(model, data, y_test):
    '''
    plot roc curve and save image
    '''
    probs = model.predict_proba(data)
    # keep probabilities for the positive outcome only
    probs = probs[:, 1]

    # calculate roc curves
    fpr, tpr, _ = roc_curve(y_test, probs)

    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.', label=type(model).__name__)

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # save the plot
    plt.savefig('./images/results/roc_curve_' + type(model).__name__ + '.png')


if __name__ == '__main__':
    # import data
    bank_data = import_data(pth='./data/bank_data.csv')

    # exporatory data analysis (eda)
    eda_data = perform_eda(bank_data)

    # feature engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        dataframe=eda_data, response='Churn')

    # model training and evaluation
    train_models(X_train=X_train,
                 X_test=X_test,
                 y_train=y_train,
                 y_test=y_test)
