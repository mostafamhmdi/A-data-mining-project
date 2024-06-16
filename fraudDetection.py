import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

from sklearn.metrics import recall_score, precision_score, confusion_matrix, accuracy_score


def preprocess_fraud(df):
    rob_scaler = RobustScaler()
    df['scaled_amount'] = rob_scaler.fit_transform(
        df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = rob_scaler.fit_transform(
        df['Time'].values.reshape(-1, 1))

    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    return df


def fraud_model(df, testsSize):
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testsSize, random_state=42)

    smote = SMOTE(sampling_strategy={1: 2000}, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Adjust n_components as needed
    pca2 = PCA(n_components=18, random_state=42)
    X_train_pca_smote = pca2.fit_transform(X_train_resampled)
    X_test_pca_smote = pca2.transform(X_test)

    LR = LogisticRegression(class_weight={0: 1, 1: 3})
    LR.fit(X_train_pca_smote, y_train_resampled)

    pred = LR.predict(X_test_pca_smote)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average='macro')
    recall = recall_score(y_test, pred, average='macro')

    conf_matrix = confusion_matrix(y_test, pred)

    # Feature importances (coefficients) for the principal components
    feature_importances = np.abs(LR.coef_[0])
    sorted_idx = np.argsort(feature_importances)
    sorted_feature_importances = feature_importances[sorted_idx]
    sorted_feature_names = [f'PC{i+1}' for i in sorted_idx]

    fig, ax = plt.subplots(2, 1, figsize=(12, 12))

    classes = ['genuine', 'fraudulent']
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=classes, yticklabels=classes, ax=ax[0])
    ax[0].set_xlabel('Predicted Labels')
    ax[0].set_ylabel('True Labels')
    ax[0].set_title('Confusion Matrix')

    ax[1].barh(sorted_feature_names, sorted_feature_importances)
    ax[1].set_title('Feature Importances (Principal Components)')
    ax[1].set_xlabel('Importance')
    ax[1].set_ylabel('Principal Component')

    result_text = (
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
    )
    ax[1].text(1.05, 0.5, result_text, transform=ax[1].transAxes, fontsize=12,
               verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()


def run_fraud(testsize):
    df = pd.read_csv('DATA/fraud-detection/creditcard.csv')

    data = preprocess_fraud(df)

    fraud_model(data, testsSize=testsize)
