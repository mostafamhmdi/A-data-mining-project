import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import recall_score, precision_score, confusion_matrix, accuracy_score


def preprocess_heart(df):
    df.drop(['id', 'thal', 'ca', 'slope'], axis=1, inplace=True)
    df = df[df['oldpeak'].isna() == False]
    df['chol'].fillna(0, inplace=True)
    value_counts = df['fbs'].value_counts()

    false_count = value_counts.get(False, 0)
    true_count = value_counts.get(True, 0)
    total_count = false_count + true_count
    total_nan = df['fbs'].isna().sum()

    false_proportion = false_count / total_count
    true_proportion = true_count / total_count

    num_false_to_fill = round(false_proportion * total_nan)
    num_true_to_fill = total_nan - num_false_to_fill

    values_to_fill = [False] * num_false_to_fill + [True] * num_true_to_fill
    np.random.shuffle(values_to_fill)

    df.loc[df['fbs'].isna(), 'fbs'] = values_to_fill

    df.dropna(axis=0, inplace=True)

    columns_to_encode = ['dataset', 'cp', 'restecg']

    df = pd.get_dummies(df, columns=columns_to_encode)
    df['sex'].replace({'Male': 0, 'Female': 1}, inplace=True)
    df['fbs'].replace({'Male': 0, 'Female': 1}, inplace=True)
    df['exang'].replace({'Male': 0, 'Female': 1}, inplace=True)

    return df


def heart_model(df, testsSize):
    X = df.drop('num', axis=1)
    y = df['num']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testsSize, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    LR = LogisticRegression()
    LR.fit(X_train, y_train)

    pred = LR.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average='macro')
    recall = recall_score(y_test, pred, average='macro')

    conf_matrix = confusion_matrix(y_test, pred)

    feature_importances = np.abs(LR.coef_[0])
    feature_names = X.columns
    sorted_idx = np.argsort(feature_importances)
    sorted_feature_importances = feature_importances[sorted_idx]
    sorted_feature_names = feature_names[sorted_idx]

    fig, ax = plt.subplots(2, 1, figsize=(12, 12))

    classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=classes, yticklabels=classes, ax=ax[0])
    ax[0].set_xlabel('Predicted Labels')
    ax[0].set_ylabel('True Labels')
    ax[0].set_title('Confusion Matrix')

    ax[1].barh(sorted_feature_names, sorted_feature_importances)
    ax[1].set_title('Feature Importances')
    ax[1].set_xlabel('Importance')
    ax[1].set_ylabel('Feature')

    result_text = (
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
    )
    ax[1].text(1.05, 0.5, result_text, transform=ax[1].transAxes, fontsize=12,
               verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()


def run_heart(testsize):
    df = pd.read_csv('DATA/heart-disease/heart_disease_uci.csv')

    data = preprocess_heart(df)

    heart_model(data, testsSize=testsize)
