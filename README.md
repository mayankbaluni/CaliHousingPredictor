## California Housing Price Prediction with Linear Regression

This repository contains a simple and efficient implementation of a Linear Regression model for predicting housing prices in California using the popular California Housing dataset.

### Overview

The code performs the following steps:

1. **Imports necessary libraries:** numpy, sklearn.datasets, sklearn.model_selection, sklearn.linear_model, sklearn.metrics
2. **Fetches the California Housing dataset:** Using `fetch_california_housing` from sklearn.datasets.
3. **Splits the data:** Splits the data into training and testing sets with 80% training data and 20% testing data using `train_test_split` from sklearn.model_selection, ensuring a random state of 42 for reproducibility.
4. **Initializes and trains Linear Regression model:** Uses `LinearRegression` from sklearn.linear_model to create a model and trains it on the training data.
5. **Predicts housing prices:** Predicts the housing prices for the testing data using the trained model.
6. **Evaluates model performance:** Calculates R2 Score and Mean Squared Error (MSE) using `r2_score` and `mean_squared_error` from sklearn.metrics to assess the model's accuracy and error.
7. **Prints results:** Displays the R2 Score and MSE values.

### Running the code

1. Clone this repository.
2. Open a terminal in the project directory.
3. Run the following command:

```
python california_housing_lr.py
```

You can install these dependencies with the following command:

pip install -r requirements.txt

This will print the R2 Score and Mean Squared Error for the model.

### Results

The R2 Score and Mean Squared Error will vary slightly due to random splitting of the data. However, you can expect to see an R2 Score around 0.57 and an MSE around 5.5.

### Interesting Inferences

- While the Linear Regression model achieves decent accuracy (R2 ~ 0.6), there's still room for improvement. Exploring other models or feature engineering could further enhance prediction accuracy.
- The MSE suggests an average error of around $5,200 in price predictions. This may be acceptable for some applications, but for others, a more precise model might be necessary.

### Further exploration

- Try different machine learning models like Random Forest or Gradient Boosting and compare their performance to the Linear Regression model.
- Experiment with feature engineering techniques like scaling or adding interaction terms to improve the model's ability to capture complex relationships between features.
- Analyze the model's coefficients to understand which features have the strongest impact on housing prices.

This project serves as a starting point for exploring housing price prediction with machine learning. Feel free to experiment further and contribute your findings!

### License

This code is licensed under the MIT License. See the `LICENSE` file for details.


