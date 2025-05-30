# Customer Churn Prediction App - built with Streamlit
This is a interactive Streamlit web app that predicts whether a customer is likely to churn or not, depending upon various features of the customer chosen by the user. Details like Internet Service, contract type and other similar features are used. The model also considers the Tenure for which the customer has stayed and his monthly charge.
The model is used is an **XGBClassifier Model** from the **XGBoost** Python library. The categorical data is encoded using one-hot-encoding, using **Scikit-learn**.

## Features of the app
1. Streamlit powered web UI for smooth visual interface and deployment
2. Encoder and model are loaded using **Joblib**
3. Interactive sidear for easy and quick input of data
4. Realtime and fast predictions

## Project files
1. **app.py** : Main streamlit file for interface
2. **churn_predictor.pkl** : Trained XGBClassifier
3. **encoder.pkl** : Trained one hot encoder
4. **requirements.txt** : Contains the dependencies required by app.py to run

## Model info
The model was trained using XGBoost. Essentially, the model is an **XGBoost Classifier**. The data was preprocessed using one hot encoding, using the **OneHotEncoder** from sklearn. Both the model and the encoder were saved using Joblib. Next, this trained model and encoder are imported into the Streamlit application and used for preprocessing entered data and prediction. Before finalizing the XGBClassiifer, a Random Forest Classifier was also trained, but the results of XGBClassifier was more promising.
In both the cases, [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle was used.

## Author
### Soumyajit Chakraborty
