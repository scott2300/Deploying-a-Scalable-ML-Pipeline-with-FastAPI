# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project uses a machine learning model to predict whether a person's annual income is greater than $50,000 based on census data. The model was trained using the Adult Census Income dataset and implemented in Python using scikit-learn. Categorical features were processed using one-hot encoding, and the target variable was converted into a binary format for classification.

The model was developed as part of the Udacity Machine Learning DevOps program and deployed using FastAPI.

## Intended Use
The purpose of this model is to demonstrate the process of building, testing, and deploying a machine learning model. The model predicts whether an individual's income is above or below $50,000 per year using demographic and employment-related information.

This model is intended for educational purposes only and should not be used to make real-world decisions related to employment, lending, insurance, or other high-impact areas.

## Training Data
The model was trained using the Adult Census Income dataset. The dataset contains information about individuals, including age, education, occupation, marital status, work class, hours worked per week, and other demographic attributes.

The target variable is salary, which indicates whether a person's annual income is greater than $50,000.

The dataset was split into training and testing sets using an 80/20 train-test split.

## Evaluation Data
The evaluation data consisted of the 20% test dataset that was held out during training. The same preprocessing steps used on the training data were applied to the test data using the saved encoder and label converter.

Model performance was evaluated on the test dataset and on individual slices of categorical features to better understand how the model performed across different groups.

## Metrics
The model was evaluated using precision, recall, and F1 score.

Results on the test dataset:

Precision: 0.7419
Recall: 0.6384
F1 Score: 0.6863

Precision measures how often positive predictions were correct. Recall measures how many of the actual positive cases were identified correctly. The F1 score provides a balance between precision and recall.

In addition to the overall metrics, performance was evaluated across categorical feature slices. These results were saved to slice_output.txt.

## Ethical Considerations
This dataset contains demographic information such as race, sex, and native country. Because of this, there is a possibility that historical biases present in the data could influence model predictions.

Before using a model like this in a real-world setting, it would be important to perform additional fairness and bias testing. Model performance should be reviewed across demographic groups to identify any potential disparities.

## Caveats and Recommendations
This model was created as part of a learning project and has several limitations.

The dataset may not fully represent current populations or economic conditions. The model was evaluated using standard classification metrics, but additional analysis could be performed to better understand fairness and model reliability.

Future improvements could include hyperparameter tuning, cross-validation, additional feature engineering, and more detailed bias analysis. Monitoring model performance over time would also be important if the model were deployed in a production environment.