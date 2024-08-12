# Titanic Survival Prediction
This project aims to predict the survival of passengers aboard the Titanic using a dataset that includes various features such as age, gender, passenger class, and more. The project involves data preprocessing, exploratory data analysis (EDA), feature engineering, and model training using logistic regression.

# Dataset
The dataset used for this project is sourced from the Titanic dataset on Kaggle. It contains information about the passengers aboard the Titanic, including features such as Pclass, Sex, Age, Fare, and whether the passenger survived.

train.csv: The training dataset, used for building and validating the model.

test.csv: The test dataset, which can be used for final evaluation.

# Data Preprocessing
The following preprocessing steps were performed on the data:

Column Removal: Removed columns like Name, Ticket, and Cabin that are not directly useful for prediction.

Handling Missing Values: Missing values in the Age column were imputed using the mean age based on Pclass. Missing values in other columns like Embarked were dropped.

Winsorization: Applied Winsorization on the Age feature to handle outliers.

Encoding Categorical Variables: Categorical features like Sex and Embarked were converted into numerical values using one-hot encoding.

Feature Engineering: Created a new feature called FamilySize by combining SibSp and Parch.

# Exploratory Data Analysis (EDA)
Several visualizations were created to understand the relationships between features and the target variable (Survived). Key insights include:

The distribution of ages among survivors and non-survivors.

The relationship between passenger class and survival rates.

The impact of gender on survival chances.

# Model Training and Evaluation
The preprocessed data was split into training and validation sets. A logistic regression model was trained on the training set, and its performance was evaluated on the validation set using various metrics:

Accuracy: Proportion of correct predictions.

Precision: Proportion of positive identifications that were actually correct.

Recall: Proportion of actual positives that were identified correctly.

F1-score: A weighted average of Precision and Recall.

# Results
The model's performance on the validation set was as follows:

Accuracy: 0.8033707865168539

Precision: 0.7297297297297297

Recall: 0.782608695652174

F1-score: 0.7552447552447552
