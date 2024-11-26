import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
import joblib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AirQualityPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Air Quality Prediction System")

        # Set theme
        self.root.set_theme("breeze")

        # Variables
        self.city_var = tk.StringVar()
        self.result_var = tk.StringVar()

        # Initialize GUI
        self.initialize_gui()

    def initialize_gui(self):
        # Create and place widgets
        ttk.Label(self.root, text="Enter City:").grid(row=0, column=0, padx=10, pady=10)
        ttk.Entry(self.root, textvariable=self.city_var).grid(row=0, column=1, padx=10, pady=10)

        ttk.Button(self.root, text="Predict", command=self.predict_air_quality).grid(row=1, column=0, columnspan=2, pady=10)

        ttk.Label(self.root, textvariable=self.result_var).grid(row=2, column=0, columnspan=2, pady=10)

    def predict_air_quality(self):
        # Replace this with your actual prediction logic using the pre-trained model
        # For example, you might load a model and predict air quality based on the input city
        # Here, we're using a dummy prediction for demonstration purposes
        # -*- coding: utf-8 -*-

        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns, numpy as np, os
        from scipy.stats import pearsonr, spearmanr

        # Load the dataset
        df = pd.read_excel(r'C:\Users\pc\Downloads\data.xlsx', parse_dates=True)
        df.set_index('datetime', inplace=True)
        # Check the first few rows of the dataset
        print(df.head())

        # Check the basic information about the dataset
        print(df.info())

        # Check the statistical summary of the dataset
        print(df.describe())

        # Check for missing values in the dataset
        print(df.isnull().sum())

        # Drop the data after 2019

        # Visualize the distribution of each variable using histograms
        df.hist(bins=50, figsize=(20, 15))
        plt.show()

        # mask = np.zeros_like(df.corr(), dtype=np.bool)
        # mask[np.triu_indices_from(mask)] = True

        # Visualize the correlations between variables using a heatmap
        sns.heatmap(df.corr(), cmap='coolwarm', vmin=-0.9, vmax=0.9, annot=True, fmt='.2f')
        plt.show()

        # Visualize the relationship between the target variable and the other variables using scatterplots
        sns.pairplot(df, x_vars=['ws', 'wd', 'temp', 'dew_temp', 'pressure', 'wv', 'blh', 'bcaod550', 'duaod550',
                                 'omaod550', 'ssaod550', 'suaod550', 'aod469', 'aod550', 'aod670', 'aod865', 'aod1240'],
                     y_vars=['pm2p5'], height=7, aspect=0.7)
        plt.show()

        # , height=8, aspect=0.5

        from sklearn.feature_selection import mutual_info_regression

        X = df.copy()
        y = X.pop("pm2p5")

        def make_mi_scores(X, y):
            mi_scores = mutual_info_regression(X, y)
            mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
            mi_scores = mi_scores.sort_values(ascending=False)
            return mi_scores

        mi_scores = make_mi_scores(X, y)
        mi_scores[::3]  # show a few features with their MI scores

        def plot_mi_scores(scores):
            scores = scores.sort_values(ascending=True)
            width = np.arange(len(scores))
            ticks = list(scores.index)
            plt.barh(width, scores)
            plt.yticks(width, ticks)
            plt.title("Mutual Information Scores")

        plt.figure(dpi=100, figsize=(8, 5))
        plot_mi_scores(mi_scores)

        # Calculate Pearson correlation coefficient
        corr_p, p_val_p = pearsonr(df['pm2p5'], df['pressure'])
        print("Pearson correlation coefficient:", corr_p)
        print("p-value:", p_val_p)

        # Calculate Spearman correlation coefficient
        corr_s, p_val_s = spearmanr(df['pm2p5'], df['pressure'])
        print("Spearman correlation coefficient:", corr_s)
        print("p-value:", p_val_s)

        """city = self.city_var.get()
        prediction = self.dummy_predict(city)

        # Display the result
        result_text = f"Predicted Air Quality in {city}: {prediction}"
        self.result_var.set(result_text)

        # Show a sample plot (replace with actual visualization logic)
        self.show_sample_plot() """

    """ def dummy_predict(self, city):
        # Dummy prediction function (replace with your actual prediction logic)
        # Here, we're generating a random number as a placeholder
        return np.random.randint(0, 101)

    def show_sample_plot(self):
        # Generate a sample plot (replace with actual visualization logic)
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2, 3, 4], [10, 30, 20, 40, 35], label='Sample Data')
        ax.set_xlabel('Time')
        ax.set_ylabel('Air Quality')
        ax.legend()

        # Embed the plot in the GUI
        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=3, column=0, columnspan=2, pady=10) """
# -*- coding: utf-8 -*-










if __name__ == "__main__":
    root = ThemedTk()
    app = AirQualityPredictionApp(root)
    root.mainloop()
