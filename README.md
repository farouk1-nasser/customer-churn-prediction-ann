# Bank Customer Churn Prediction (ANN)

An Artificial Neural Network built with TensorFlow/Keras to predict whether a bank customer will leave (churn), using the Churn Modelling dataset.

## Dataset

The notebook uses `Churn_Modelling.csv` (10,000 customers). Download it from Kaggle ("Churn Modelling" dataset) and update the `path` variable in the notebook to point to your copy.

## Workflow

1. Load data and drop non-informative columns (`RowNumber`, `CustomerId`, `Surname`, `Gender`)
2. One-hot encode `Geography`
3. Split into train / validation / test (70% / 15% / 15%, stratified)
4. Standardize features with `StandardScaler`
5. Train the ANN and evaluate with a confusion matrix

## Model Architecture

```
Dense(13, ReLU)
Dense(32, ReLU) -> Dropout(0.2)
Dense(64, ReLU) -> Dropout(0.2)
Dense(1, Sigmoid)
```

- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Batch size:** 32

## Results

Validation accuracy of about **85-86%** (see the notebook for training curves and the confusion matrix).

## Tech Stack

Python, TensorFlow/Keras, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## How to Run

```bash
git clone https://github.com/farouk1-nasser/customer-churn-prediction-ann.git
cd customer-churn-prediction-ann
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
jupyter notebook churn_ann.ipynb
```

Or open the notebook in Google Colab.

## Author

**[Farouk Nasser](https://github.com/farouk1-nasser)**
Electronics & Communications Engineering student | Data Science & Machine Learning
