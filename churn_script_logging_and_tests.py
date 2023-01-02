'''
Testing and logging module for churn_library churn prediction
Author: Paul Hake
Date: 12/31/22

'''
import os
import logging
import churn_library as cls

# setup logger configuration
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the     other test functions
    '''
    # check the import file exists
    try:
        data = cls.import_data("./data/bank_data.csv")
        logging.info("test_import: SUCCESS")
    except FileNotFoundError as err:
        logging.error("test_import: The file wasn't found")
        raise err

    # check the data file has rows and columns
    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    eda_data = cls.import_data('./data/bank_data.csv')
    try:
        cls.perform_eda(data=eda_data)
        logging.info('perform_eda: SUCCESS')
    except KeyError as err:
        logging.error('column "%s" not found', err.args[0])
        raise err

    try:
        # check for nulls
        assert sum(eda_data.isnull().sum()) == 0
    except AssertionError as err:
        logging.warning('WARNING: There are nulls in the data file')
        raise err
    try:
        # check that eda files were created
        assert os.path.exists('./data/nulls_check.csv')
        assert os.path.exists('./images/eda/Churn_hist.png')
        assert os.path.exists('./images/eda/customer_age_hist.png')
        assert os.path.exists('./images/eda/data_correlation_heatmap.png')
        assert os.path.exists('./images/eda/marital_status_hist.png')
        assert os.path.exists('./images/eda/Total_Trans_Ct_distribution.png')
        logging.info('perform_eda: SUCCESS - checking for eda result files')
    except AssertionError as err:
        logging.warning('WARNING: some or all eda result files not created')
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''

    # Load the DataFrame
    data = cls.import_data("./data/bank_data.csv")
    number_of_original_columns = data.shape[1]

    # add churn feature
    data['Churn'] = data['Attrition_Flag'].\
        apply(lambda val: 0 if val == "Existing Customer" else 1)

    try:
        (X_train, X_test, _, _) = cls.perform_feature_engineering(
            dataframe=data,
            response='Churn')

        # check for churn column
        assert 'Churn' in data.columns
        logging.info(
            "Testing perform_feature_engineering for Churn column: SUCCESS")
    except KeyError as err:
        logging.error('Churn column missing: ERROR')
        raise err

    try:
        # check that new columns created or removed by feature engineering
        new_columns_created = X_train.shape[1]
        assert new_columns_created != number_of_original_columns
        logging.info(
            '%s number of columns before feature engineering ',
            number_of_original_columns)
        logging.info(
            '%s number of columns after feature engineering ',
            new_columns_created)
    except AssertionError as err:
        logging.error(
            'perform_feature_engineering: same number of features before and after')
        raise err

    logging.info(
        'perform_eda > Train data shape is %s, Test data shape is %s',
        X_train.shape,
        X_test.shape)

    try:
        train_test_split = round(
            (X_train.shape[0] / (X_train.shape[0] + X_test.shape[0])), 2)
        assert train_test_split == 0.7
        logging.info('train test split is %s', train_test_split)
    except AssertionError as err:
        logging.error(
            "perform_eda train test split incorrect: train proportion is %s",
            train_test_split)
        raise err


def test_train_models():
    '''
    test train_models
    '''
    # Load the DataFrame
    data = cls.import_data("./data/bank_data.csv")

    # add churn feature
    data['Churn'] = data['Attrition_Flag'].\
        apply(lambda val: 0 if val == "Existing Customer" else 1)

    (X_train, X_test, y_train, y_test) = cls.perform_feature_engineering(
        dataframe=data,
        response='Churn')
    # check model files created
    try:
        cls.train_models(X_train, X_test, y_train, y_test)
        # check for saved model files
        assert os.path.exists('./models/logistic_model.pkl')
        assert os.path.exists('./models/rfc_model.pkl')
        logging.info('train_models - lr and rf model files saved: SUCCESS')
    except AssertionError as err:
        logging.error('ERROR: one or both model files not created')
        raise err

    try:
        # check for results files
        logging.info('train_models: checking for results files')
        assert os.path.exists('./data/y_test_preds_lr.csv')
        assert os.path.exists('./data/y_train_preds_lr.csv')
        assert os.path.exists('./data/y_test_preds_rf.csv')
        assert os.path.exists('./data/y_train_preds_rf.csv')
        logging.info('test_train_models ran: SUCCESS')
    except AssertionError as err:
        logging.error('ERROR: train_models not all results files created')
        raise err

    try:
        # check for classification report image files
        assert os.path.exists(
            './images/results/classification_report_logistic_regression.png')
        assert os.path.exists(
            './images/results/classification_report_random_forest.png')
        logging.info('train_models - 2 classification reports found: SUCCESS')
    except AssertionError as err:
        logging.error('train_models - classification report(s) missing')
        raise err

    try:
        # check for feature importance image file
        assert os.path.exists('./images/results/cv_rfc_model_feature_importance.png')
        logging.info('train_models - feature importance report found: SUCCESS')
    except AssertionError as err:
        logging.error('train_models - feature importance report missing')
        raise err

    try:
        # check roc curve reports were created
        assert os.path.exists('./images/results/roc_curve_LogisticRegression.png')
        logging.info(
            'train_models - logistic regression roc report found: SUCCESS')
    except AssertionError as err:
        logging.error('train_models - logistic regression roc report missing')
        raise err

    try:
        # check random forest curve report was created
        assert os.path.exists('./images/results/roc_curve_RandomForestClassifier.png')
        logging.info('train_models - random forest roc report found: SUCCESS')
    except AssertionError as err:
        logging.error('train_models - random forest roc report missing')
        raise err


if __name__ == "__main__":
    # testing modules if ran via command line
    test_import()
    test_eda()
    test_perform_feature_engineering()
    test_train_models()
