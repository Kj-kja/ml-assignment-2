# Breast Cancer Classification Using Machine Learning

## 1. Project Overview
This project compares five machine learning classification algorithms on the Breast Cancer Wisconsin Diagnostic dataset. The goal is to classify tumours as malignant or benign and evaluate the performance of different models.

## 2. Dataset
- **Dataset:** Breast Cancer Wisconsin Diagnostic Dataset
- **Number of records:** 569
- **Number of input features:** 30
- **Target:** Malignant (0) or Benign (1)
- **Train-test split:** 80% training and 20% testing

## 3. Machine Learning Models
The following classification algorithms were implemented:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 4. Evaluation Metrics
The models were evaluated using:
- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)

## 5. Results

| Model | Accuracy | AUC |
|---|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 |
| Decision Tree | 0.9123 | 0.9157 |
| KNN | 0.9561 | 0.9788 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 |
| SVM | 0.9825 | 0.9950 |

Logistic Regression and SVM achieved the highest test accuracy of 98.25%. Logistic Regression achieved the highest AUC of 0.9954 among the evaluated models.

## 6. Streamlit Application
The Streamlit application allows users to upload test data, select a classification model, and view evaluation metrics and classification results.

## 7. Project Files
- `ML_Assignment_2.ipynb` — model training and evaluation notebook
- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `test_data.csv` — held-out test dataset
- `model_comparison.csv` — model evaluation results
- Model `.pkl` files — saved trained models and preprocessing scaler

## 8. How to Run
Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 9. Conclusion
This project demonstrates the implementation and comparison of five machine learning classification algorithms. Logistic Regression and SVM performed best in terms of test accuracy, while Logistic Regression achieved the highest AUC in this experiment.
