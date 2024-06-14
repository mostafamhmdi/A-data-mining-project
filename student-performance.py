import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import root_mean_squared_error as rmse

def preprocess_students(data):
    data['school'] = data['school'].map({'GP': 0, 'MS': 1})
    data['sex'] = data['sex'].map({'F': 0, 'M': 1})
    data['address'] = data['address'].map({'U': 0, 'R': 1})
    data['famsize'] = data['famsize'].map({'LE3': 2, 'GT3': 4})
    data['Pstatus'] = data['Pstatus'].map({'T': 1, 'A': 0})
    data['schoolsup'] = data['schoolsup'].map({'yes': 1, 'no': 0})
    data['famsup'] = data['famsup'].map({'yes': 1, 'no': 0})
    data['paid'] = data['paid'].map({'yes': 1, 'no': 0})
    data['activities'] = data['activities'].map({'yes': 1, 'no': 0})
    data['nursery'] = data['nursery'].map({'yes': 1, 'no': 0})
    data['internet'] = data['internet'].map({'yes': 1, 'no': 0})
    data['higher'] = data['higher'].map({'yes': 1, 'no': 0})
    data['romantic'] = data['romantic'].map({'yes': 1, 'no': 0})
    columns_to_encode = ['Mjob', 'Fjob', 'reason', 'guardian']

    data = pd.get_dummies(data, columns=columns_to_encode)
    return data


def student_model(data, testsSize):
    X = data.drop(['G3', 'G1'], axis=1)
    y = data['G3']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    selected_features = []
    remaining_features = list(range(X_train.shape[1]))
    best_score = -np.inf
    feature_names = X.columns

    while remaining_features:
        scores = []
        for feature in remaining_features:
            features_to_try = selected_features + [feature]
            X_train_subset = X_train.iloc[:, features_to_try]

            LR = LinearRegression()
            cv_scores = cross_val_score(
                LR, X_train_subset, y_train, cv=5, scoring='neg_mean_squared_error')
            mean_score = np.mean(cv_scores)
            scores.append((mean_score, feature))

        scores.sort(reverse=True, key=lambda x: x[0])
        best_new_score, best_new_feature = scores[0]

        if best_new_score > best_score:
            selected_features.append(best_new_feature)
            remaining_features.remove(best_new_feature)
            best_score = best_new_score
            print(
                f"Selected feature: {feature_names[best_new_feature]}, Cross-validated score: {best_new_score:.4f}")
        else:
            break

    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]
    final_LR = LinearRegression()
    final_LR.fit(X_train_selected, y_train)

    test_score = final_LR.score(X_test_selected, y_test)

    print(
        f"Final selected features: {[feature_names[i] for i in selected_features]}")
    
    pred = final_LR.predict(X_test_selected)
    print("MSE:    ",mse(y_test, pred))
    print("RMSE:    ",rmse(y_test, pred))
    print("R2Score:    ",r2_score(y_test, pred))
    


def run_student():
    df = pd.read_csv('DATA/student-performance/student.csv')

    data = preprocess_students(df)

    student_model(data)
s