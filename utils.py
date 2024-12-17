import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

def scale_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X = []
    y = []

    sequence_length = 1

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, :-1])
        y.append(scaled_data[i, -1])

    X, y = np.array(X), np.array(y)

    return X, y, scaler 

def inverse_transform(scaler, predictions):
    predictions = predictions.reshape(-1, 1)
    inverse_predictions = scaler.inverse_transform(np.concatenate([np.zeros((len(predictions), scaler.n_features_in_ - 1)), predictions], axis=1))
    return inverse_predictions[:, -1]

def lineplot(y_test, y_pred, y_pred_new=None):
    """
    Membuat plot data asli (y_test), prediksi (y_pred), 
    dan prediksi untuk data baru (y_pred_new) jika tersedia.
    
    Args:
        y_test (array-like): Data asli (data uji).
        y_pred (array-like): Prediksi untuk data uji.
        y_pred_new (array-like, optional): Prediksi untuk data baru. Default None.
    """
    # Plot data asli (y_test) dan prediksi (y_pred)
    plt.plot(y_test, color='blue', label='Real Close Price')
    plt.plot(y_pred, color='red', label='Predicted Close Price')

    # Plot prediksi untuk data baru jika diberikan
    if y_pred_new is not None:
        plt.plot(range(len(y_test), len(y_test) + len(y_pred_new)), y_pred_new, 
                 color='green', linestyle='--', label='New Data Prediction')

    plt.title('Close Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def scatterplot(y_test, y_pred):
    # Membuat plot scatter
    plt.scatter(range(len(y_pred)), y_pred, label='Predicted', color='red', alpha=0.5)
    plt.scatter(range(len(y_test)), y_test, label='Actual', color='blue', alpha=0.5)
    plt.title('Comparison between Predicted and Actual Values')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()