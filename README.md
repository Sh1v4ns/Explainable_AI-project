This repository hosts the implementation of ViaSHAP, a novel approach that learns a function to compute Shapley values, from which the predictions can be derived directly by summation. We explore two learning approaches based on the universal approximation theorem and the Kolmogorov-Arnold representation theorem. ViaSHAP using Kolmogorov-Arnold Networks performs on par with state-of-the-art algorithms for tabular data. The explanations obtained using ViaSHAP are significantly more accurate than other popular approximators, e.g., FastSHAP on both tabular data and images. All the experiments have been conducted in a Python 3.10 environment.

We have implemented 3 different approaches MLP ViaSHAP , KAN ViaSHAp , ChebKAN ViaSHap the implementation details and results are below 

# MLP ViaSHAP implementation:
Usage:
Clone the Repository: Clone this repository to your local machine using the following command:

git clone https://github.com/Sh1v4ns/XAI_Project.git
Install Dependencies: Ensure that you have the necessary dependencies installed. You can install them using pip:

pip install -r requirements.txt
We have used the MLP model for our implementation

The XAI method used is ViaSHAP (mlpshap.py)

Dataset used is Elevators Dataset(tabular regression)

Results reproduced :

Prediction performance : RMSE / Accuracy

SHAP quality metric : Cosine similarity with true Shapley values

Feature importance ranking :Compare important features

Also includes : AUC / performance and similarity to true Shapley values

# KANSHAP & CHEBKAN XAI Implementation
Usage
1. Clone the Repository

Clone this repository to your local machine using the following command:

git clone https://github.com/your-username/your-repo-name.git
2. Install Dependencies

Ensure that you have the necessary dependencies installed. You can install them using:

pip install -r requirements.txt
KANSHAP Implementation
3. Model Used

We use the KAN (Kolmogorov-Arnold Network) model for our implementation.

4. XAI Method

The explainability method used is KANSHAP (kanshap.py).

5. Dataset

Dataset used is Elevators Dataset (tabular regression).

6. Results Reproduced

i. Prediction Performance

RMSE / Accuracy

ii. SHAP Quality Metric

Cosine similarity with true Shapley values

iii. Feature Importance Ranking

Comparison of important features

iv. Additional Metrics

AUC / Performance and similarity to true Shapley values
CHEBKAN Implementation
7. Model Used

We use the Chebyshev Kolmogorov-Arnold Network (CHEBKAN) model for this implementation.

8. XAI Method

The explainability method used is CHEBKAN-SHAP (chebkan_shap.py).

9. Dataset

Dataset used is Elevators Dataset (tabular regression).

10. Results Reproduced

i Time comparison of ChebKAN with ViaShap

ii. Feature Importance Ranking

Compare important features across methods

iii. Additional Metrics

AUC / Performance and similarity to true Shapley values

