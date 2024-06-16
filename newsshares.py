import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import root_mean_squared_error as rmse
from scipy.stats import zscore


def preprocess_news(df):
    df.drop(['url', ' timedelta'], inplace=True, axis=1)

    data = df[df[' n_tokens_content'] < 4000]
    data = data[data[' n_unique_tokens'] < 100]
    data = data[data[' num_hrefs'] < 100]
    data = data[data[' num_imgs'] < 60]
    data = data[data[' num_videos'] < 40]
    data = data[data[' kw_min_min'] < 40]
    data = data[data[' kw_max_min'] < 100000]
    data = data[data[' kw_avg_min'] < 20000]
    data = data[data[' kw_avg_max'] < 600000]
    data = data[data[' kw_min_max'] < 200000]
    data = data[data[' kw_max_avg'] < 150000]
    data = data[data[' kw_avg_avg'] < 20000]
    data = data[data[' self_reference_min_shares'] < 200000]
    data = data[data[' self_reference_max_shares'] < 200000]
    data = data[data[' self_reference_avg_sharess'] < 200000]

    features_to_transform = [' kw_max_min', ' kw_avg_min', ' kw_min_max', ' kw_max_max',
                             ' kw_max_avg', ' kw_avg_avg', ' self_reference_min_shares',
                             ' self_reference_max_shares', ' self_reference_avg_sharess', ' shares']

    for feature in features_to_transform:
        if (data[feature] <= 0).any():
            data[feature +
                 '_log'] = np.log1p(data[feature] - data[feature].min() + 1)
        else:
            data[feature + '_log'] = np.log1p(data[feature])

    features = [' n_unique_tokens', ' n_non_stop_unique_tokens', ' average_token_length', ' global_subjectivity', ' global_sentiment_polarity', ' avg_positive_polarity',
                ' title_sentiment_polarity', ' kw_max_min_log', ' kw_avg_min_log', ' kw_avg_avg_log', ' self_reference_min_shares_log', ' self_reference_avg_sharess_log', ' shares_log']
    z_scores = np.abs(data[features].apply(zscore))

    threshold = 3

    outliers = (z_scores > threshold).any(axis=1)

    final_df = data[~outliers]
    final_df.columns = final_df.columns.str.strip()
    final_df.drop(['n_tokens_title', 'n_unique_tokens',
                   'n_non_stop_words', 'n_non_stop_unique_tokens', 'kw_max_max_log'], axis=1, inplace=True)
    return final_df


def news_model(data, testsSize):
    model_df = data[['kw_avg_avg_log', 'is_weekend', 'data_channel_is_tech', 'data_channel_is_socmed', 'self_reference_min_shares_log', 'kw_min_max_log', 'kw_min_avg', 'LDA_00', 'n_tokens_content', 'data_channel_is_entertainment', 'global_subjectivity', 'kw_max_avg_log', 'kw_avg_max', 'kw_min_min',
                     'min_positive_polarity', 'num_self_hrefs', 'num_hrefs', 'weekday_is_friday', 'weekday_is_monday', 'title_sentiment_polarity', 'kw_max_min_log', 'average_token_length', 'avg_positive_polarity', 'abs_title_sentiment_polarity', 'title_subjectivity', 'abs_title_subjectivity', 'shares_log']]
    X = model_df.drop(['shares_log'], axis=1)
    y = model_df['shares_log']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    LR = LinearRegression()
    LR.fit(X_train, y_train)
    y_pred_log = LR.predict(X_test)

    y_pred = np.expm1(y_pred_log)
    y_test_unlog = np.expm1(y_test)

    print("Mean Squared Error (MSE):", mse(y_test, y_pred_log))
    print("Mean Squared Error (MSE):", mse(y_test_unlog, y_pred))


def run_news():
    df = pd.read_csv('DATA/news-popularity/OnlineNewsPopularity.csv')

    data = preprocess_news(df)

    news_model(data)
