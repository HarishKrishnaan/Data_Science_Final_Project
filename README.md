# Vending Machine Sale Predictions

This repository uses machine learning models such as Random Forest Classification, Logistic Regression, and Decision Tree Classification to identify vending machine item attributes that contribute to successful sales.

## Project Goal

The goal of this project is to predict whether a vending machine item will be selected based on both product attributes and placement attributes.

Product attributes include:
- Item size
- Item type

Placement attributes include:
- Edge placement
- Eye-level placement
- Distance from the center of the machine
- Row and column position

## Models

The following machine learning models were evaluated:

- Random Forest Classifier
- Logistic Regression
- Decision Tree Classifier

## Evaluation Metrics

The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Mean predicted probability

## Repository Structure

```text
data/
    Survey Results.csv
    Machine 1.csv
    Machine 2.csv
    Machine 3.csv
    Machine 4.csv
    Machine 5.csv
    item_type.csv

figures/
    Accuracy and mean probabilty plot
    Confusion matrices
    Feature importance plots
    Probability distributions

results/
    Hyper Parameter Metrics csv
    Model Metrics csv

main.py
requirements.txt
README.md
```

# Installation
Install required dependencies
```bash
pip install -r requirements.txt
```

# Running the Project
Run the main program:
```bash
python main.py
```

# Reproducing Results
The repository contains:

- Data preprocessing code
- Model training code
- Evaluation functions
- Visualization generation

Running `main.py` will reproduce the experiments, evaluation metrics, and visualizations used in the final report.

# Dataset Notes

The dataset was generated from vending machine survey responses. Each observation represents an item exposure row containing both item attributes and vending machine placement information. The target variable indicates whether the item was selected by a respondent.

Authors
* Harish Krishnan
* Johnson Nguyen