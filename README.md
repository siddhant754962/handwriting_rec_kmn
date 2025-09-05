

# **Handwritten Digit Recognition using KNN**

## **Overview**

This project implements **handwritten digit recognition** using the **K-Nearest Neighbors (KNN)** algorithm on the **MNIST dataset**. The dataset contains **70,000 grayscale images** of handwritten digits (0–9), each **28x28 pixels**.

The goal is to classify digits based on pixel values using **classical machine learning**, demonstrating ML pipelines, data preprocessing, and evaluation.

---

## **Technologies**

* Python
* Pandas & NumPy
* Scikit-learn (KNN, StandardScaler, Pipeline, PCA)
* Matplotlib & Seaborn (EDA & visualization)
* Joblib (saving trained model)

---

## **Key Features**

* Full **data exploration** and visualization of MNIST digits
* **KNN pipeline** with optional **PCA** for faster computation
* **Evaluation** using accuracy, confusion matrix, and classification report
* Trained model saved for **reuse and deployment**

---

## **Results**

* **Accuracy:** \~95–97% on MNIST test set
* Visualizations highlight **correct and misclassified digits**
* Demonstrates **classical ML workflow** on image data

---

## **Limitations**

* KNN is limited to **MNIST-like preprocessed data**
* Cannot reliably predict **hand-drawn digits outside the dataset**
* No interactive **drawing app** implemented

---

## **Future Improvements**

* Upgrade to **CNN for higher accuracy (\~99%)**
* Add a **Streamlit drawing app** for live digit recognition
* Display **top-k predictions with confidence scores**

---

## **How to Run**

1. Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

2. Place `mnist_train.csv` and `mnist_test.csv` in the project folder.
3. Run the script:

```bash
python knn_mnist_pipeline.py
```

4. The trained pipeline will be saved as `knn_mnist_pipeline.pkl`.

---



---

