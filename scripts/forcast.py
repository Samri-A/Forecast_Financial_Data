import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class forcast_port:
    def __init__(self , model_path):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

    def predict(self, csv_path , months):
        df = pd.read_csv(csv_path)
        scaler = MinMaxScaler()
        df['Close'] = scaler.fit_transform(df[['Close']])
        data = df["Close"]
        X = []
        for i in range(len(data) - 120 + 1):
            row = data[i:i + 120]            
            X.append(row)
        X = np.array(X)
        predictions = []
        current_sequence = X[-1:].copy()  
        
        for i in range(months):
            pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(pred)
            current_sequence = np.append(current_sequence[:, -119:], pred, axis=1)
        
        predictions = np.array(predictions)
        predictions = predictions.reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)

        return predictions

    def plot_forcast(self, first_date ,  predictions ,periods , csv_path):
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"])
        forcast_index = pd.date_range(start=first_date, periods=periods, freq='D')
        plt.figure(figsize=(12, 6))
        plt.plot(forcast_index , predictions, label='Predicted' , color='orange')
        plt.plot(df["Date"] , df["Close"], label='History' , color='blue')
        plt.title('Histoory and Forcast')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

