import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


def scale_data(df, subset_columns=None, target_column=None, sequence_length=1):
    if subset_columns is None or target_column is None:
        df_subset = df
    else:
        df_subset = df[subset_columns + [target_column]]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_subset)

    X = []
    y = []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, :-1])  
        y.append(scaled_data[i, -1])

    X, y = np.array(X), np.array(y)

    return X, y, scaler


def inverse_transform(scaler, predictions, model_type='default'):
    predictions_2d = predictions.reshape(-1, 1)

    if model_type == 'lstm':
        dummy_data = np.zeros((len(predictions_2d), scaler.n_features_in_ - 1))
        inverse_predictions = scaler.inverse_transform(np.concatenate([dummy_data, predictions_2d], axis=1))
    else:
        inverse_predictions = scaler.inverse_transform(np.concatenate([np.zeros((len(predictions_2d), scaler.n_features_in_ - 1)), predictions_2d], axis=1))

    return inverse_predictions[:, -1]


def lineplot(y_test, y_pred, y_pred_new=None):
    plt.plot(y_test, color='blue', label='Real Close Price')
    plt.plot(y_pred, color='red', label='Predicted Close Price')

    if y_pred_new is not None:
        plt.plot(range(len(y_test), len(y_test) + len(y_pred_new)), y_pred_new, 
                 color='green', linestyle='--', label='New Data Prediction')

    plt.title('Close Price Prediction vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def scatterplot(y_test, y_pred):
    plt.scatter(range(len(y_pred)), y_pred, label='Predicted', color='red', alpha=0.5)
    plt.scatter(range(len(y_test)), y_test, label='Actual', color='blue', alpha=0.5)
    plt.title('Comparison between Predicted and Actual Values')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()