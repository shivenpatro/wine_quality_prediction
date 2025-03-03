# Wine Quality Prediction Project

## Overview

This project focuses on predicting the quality of red wine using machine learning techniques. By analyzing various physicochemical properties of red wine, such as acidity, sugar content, and alcohol level, we aim to build a predictive model that can accurately assess wine quality. This project is designed to be educational and provides a clear, step-by-step guide for understanding and implementing a machine learning workflow for wine quality prediction.

## Dataset

The dataset used in this project is the 'Wine Quality Dataset', specifically focusing on red wine. It contains various features describing the physicochemical properties of wine samples, along with a quality score rated by experts.

### Dataset Files:
- `winequality-red.csv`: Contains the data for red wine quality prediction.
- `WineQualityPrediction.ipynb`: Jupyter Notebook containing the code and explanations for the project.

### Features:
The dataset includes the following features:
- **Fixed acidity**:  Acids that are non-volatile.
- **Volatile acidity**:  The amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste.
- **Citric acid**:  Found in citrus fruits, citric acid can add 'freshness' and flavor to wines.
- **Residual sugar**: The amount of sugar remaining after fermentation, it's rare to find wines with less than 1 gram/liter and wines with greater than 45 grams/liter are considered sweet.
- **Chlorides**: The amount of salt in the wine.
- **Free sulfur dioxide**: The free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents microbial growth and the oxidation of wine.
- **Total sulfur dioxide**: Amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2 concentrations over 50 ppm, SO2 becomes evident in the nose and taste of wine.
- **Density**: The density of wine is close to that of water depending on the alcohol and sugar content.
- **pH**: Describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 pH.
- **Sulphates**: A wine additive which can contribute to sulfur dioxide gas (SO2) levels, which acts as an antimicrobial and antioxidant.
- **Alcohol**: The percent alcohol content of the wine.
- **Quality**: Output variable (score between 3 and 8).

## Project Structure

```
wine_quality_prediction/
├── README.md              # This README file
├── winequality-red.csv    # Dataset file
└── WineQualityPrediction.ipynb # Jupyter Notebook with project code and explanations
```

## How to Run the Project

To run this project, you will need to have Python and Jupyter Notebook installed on your system. Additionally, you will need to install the required Python libraries.

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required Python Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - imbalanced-learn

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### Steps to Run

1. **Clone the repository** (if you have uploaded this to GitHub):
   ```bash
   git clone [repository-url]
   cd wine_quality_prediction
   ```

2. **Open the Jupyter Notebook:**
   Navigate to the `wine_quality_prediction` directory and open the `WineQualityPrediction.ipynb` file using Jupyter Notebook.
   ```bash
   jupyter notebook WineQualityPrediction.ipynb
   ```

3. **Run the Notebook Cells:**
   Once the Jupyter Notebook is open, you can execute each cell sequentially by selecting a cell and clicking on the "Run" button or pressing `Shift + Enter`. Follow the notebook from top to bottom to understand the data analysis, model building, and evaluation process.

## Project Overview in `WineQualityPrediction.ipynb`

The Jupyter Notebook `WineQualityPrediction.ipynb` is structured to guide you through the wine quality prediction project step-by-step:

1. **Introduction**: Provides an overview of the project and objectives.
2. **Dataset Overview**: Loads the dataset and explores its basic properties, including shape, information, description, and missing values.
3. **Data Preprocessing**: Prepares the data for model training, including handling class imbalance using SMOTE, splitting data into training and testing sets, and scaling features.
4. **Model Training**: Defines the hyperparameter space and trains a Gradient Boosting Classifier model using RandomizedSearchCV for hyperparameter tuning.
5. **Model Evaluation**: Evaluates the trained model's performance using accuracy, classification report, and confusion matrix. Visualizes the confusion matrix.
6. **Feature Importance**:  Analyzes and visualizes feature importance to understand which physicochemical properties are most influential in predicting wine quality.
7. **Conclusion**: Summarizes the project findings and key insights.

## Conclusion

This project provides a comprehensive guide to building a machine learning model for wine quality prediction. By following the steps in the Jupyter Notebook and referring to this README, you can gain a solid understanding of the data science process, from data exploration and preprocessing to model training and evaluation. This project is suitable for individuals interested in learning about machine learning, data analysis, and wine quality assessment.
