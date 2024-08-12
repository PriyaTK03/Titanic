#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


# Define the path to your CSV file
train_data = pd.read_csv(r"C:\Users\priya\Documents\Titanic\Dataset\train.csv")
test_data = pd.read_csv(r"C:\Users\priya\Documents\Titanic\Dataset\test.csv")

#exploring the data - .head() represents only first five rowa of data
print(train_data.head())
print(train_data.tail())
print(train_data.info())

#removing columns
column_to_remove = ['Name', 'Ticket', 'Cabin']
train_data.drop(column_to_remove, axis=1, inplace = True)
print(train_data.head(8))
#checking for missing values
#to count missing values in each column.
missing_counts = train_data.isnull().sum()
#to count missing values in the entire dataset.
missing_counts_entire = train_data.isnull().sum().sum()
print(missing_counts)
print(missing_counts_entire)


average_age_by_pclass = train_data.groupby('Pclass')['Age'].mean()

def impute_age(row):
  if pd.isna(row['Age']):  # Check if 'Age' is missing
    pclass = row['Pclass']
    # Impute with class average, handle potential NaN in average_age_by_pclass
    return average_age_by_pclass.get(pclass)  # Use get method to avoid KeyError
  else:
    return row['Age']  # Keep existing value if 'Age' is not missing
train_data['Age'] = train_data.apply(impute_age, axis=1)

train_data = train_data.dropna(axis = 0)
print(train_data.isnull().sum())

# Calculate IQR (Interquartile Range)
Q1 = train_data['Age'].quantile(0.3333)
Q3 = train_data['Age'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds for winsorization (typically 1.5 * IQR from quartiles)
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Winsorize the 'Age' feature
train_data['Age_winsorized'] = train_data['Age'].clip(lower=lower_bound, upper=upper_bound)

#Explore the winsorized data (optional)
print(train_data[['Age', 'Age_winsorized']].head())  # This shows the original and winsorized 'Age' values for the first few rows

#visualizing
print('\nMinimum age', train_data['Age_winsorized'].min())
print('Maximum age', train_data['Age_winsorized'].max())
sns.boxplot(x = train_data['Survived'], y = train_data['Age_winsorized'])
plt.show()

print('Number of Men and Women:')
print(train_data['Sex'].value_counts())
sns.boxplot(x = train_data['Age'], y = train_data['Sex'])
plt.show()

sns.barplot(x = train_data["Embarked"], y = train_data["Survived"])  
plt.show()

#converting categorical values into numerical values
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True, dtype=int)
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
train_data.drop(['SibSp', 'Parch'], axis=1, inplace = True)
print(train_data.head(8))

#Select only numerical columns for correlation calculation
numerical_columns = train_data.select_dtypes(include=['int64', 'float64']).columns
correlation = train_data[numerical_columns].corr()
#Display the correlation matrix
print(correlation)
print("\n")

plt.subplots(figsize = (15,10))
sns.heatmap(train_data.corr(), annot=True,cmap="RdBu")
plt.title("Correlations Among Features", fontsize = 18)
plt.show()

X_train = train_data.drop(columns=["Survived"])
y_train = train_data["Survived"]

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("\ny_train shape:", y_train.shape)
print("\nX_val shape:", X_val.shape)
print("\ny_val shape:", y_val.shape)
print("\n")

#Scale the Data (Scaling the input features can help improve the convergence of the optimization algorithm).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

#Model training
model = LogisticRegression(max_iter = 1000)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Calculate evaluation metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)