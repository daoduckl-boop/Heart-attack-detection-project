# Heart Attack Prediction Project

[![R](https://img.shields.io/badge/R-4.4.2-blue)](https://www.r-project.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Classification-green)](https://en.wikipedia.org/wiki/Statistical_classification)
[![Dataset](https://img.shields.io/badge/Dataset-237K_Patients-orange)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Overview

This project develops machine learning models to predict the likelihood of heart attack risk in patients based on demographic, lifestyle, and medical data. Using a dataset of **237,630 patient records**, we implemented and compared two classification models:

1. **Logistic Regression** - High accuracy & interpretability
2. **Random Forest** - High sensitivity & discrimination ability

Both models achieved strong performance with **AUC scores above 0.87**, making them valuable tools for early risk detection and clinical screening.

## Problem Statement

Cardiovascular disease remains one of the leading causes of death globally. The challenge is to:

- **Identify high-risk patients** before critical heart events occur
- **Predict heart attack likelihood** using available medical data
- **Balance sensitivity and specificity** to minimize both missed cases and false alarms
- **Provide interpretable results** for clinical decision-making

### Why This Matters

- **5.5% of patients** in the dataset had a history of heart attack
- **Early detection** can enable preventive interventions
- **Clinical actionability** requires reliable predictions
- **Resource optimization** through targeted screening

## Dataset Overview

### Size & Scope
- **Total Records**: 237,630 patient records
- **Features**: 15 variables (12 predictors + target)
- **Geographical**: Multi-state patient population (US)
- **Missing Values**: None (clean dataset)

### Key Features

| Feature | Type | Values | Description |
|---------|------|--------|-------------|
| **Age Category** | Categorical | 18-24 to 80+ | Age group in 5-year intervals |
| **Sex** | Categorical | Male, Female | Patient gender |
| **BMI** | Numerical | 12.02 - 97.65 | Body Mass Index |
| **General Health** | Categorical | Excellent to Poor | Self-reported health status |
| **Smoking Status** | Categorical | Never/Former/Current | Tobacco use history |
| **Had Diabetes** | Categorical | Yes, No | Diabetes history |
| **Had Angina** | Binary | 0, 1 | Prior angina diagnosis |
| **Had Stroke** | Binary | 0, 1 | Prior stroke history |
| **Chest Scan** | Binary | 0, 1 | Prior chest CT/scan |
| **Race/Ethnicity** | Categorical | 5 categories | Demographic classification |

### Target Variable
- **HadHeartAttack**: Binary (0 = No, 1 = Yes)
  - Class Distribution: 94.4% No, 5.5% Yes (imbalanced)

## Methodology

### Data Cleaning & Preprocessing
- ✅ Verified no missing values
- ✅ Converted categorical variables to factors
- ✅ Handled class imbalance through stratified sampling
- ✅ Exploratory Data Analysis (EDA) on all features

### Exploratory Data Analysis (EDA)

Key findings from univariate & bivariate analysis:

**By Gender**
- Males show higher heart attack rate than females
- Gender is a significant risk factor

**By Age**
- Heart attack risk increases dramatically with age
- Age 80+ has the highest prevalence
- Clear positive correlation

**By Smoking Status**
- Current smokers (especially daily) have elevated risk
- Former smokers show intermediate risk
- Never smokers have lowest risk
- Aligns with cardiovascular disease literature

**By Diabetes Status**
- Patients with diabetes show significantly higher risk
- Strong predictor of heart attack likelihood

**By General Health**
- Self-reported "Poor" health has highest risk
- Risk decreases linearly with better health status

### Model Development

#### 1. Logistic Regression
```r
glm_model <- glm(HadHeartAttack ~ ., 
                 data = train_data,
                 family = binomial(link = "logit"))
```

**Strengths:**
- Interpretable coefficients
- Fast training & prediction
- Probabilistic output
- Works well with linear relationships

#### 2. Random Forest
```r
rf_model <- randomForest(HadHeartAttack ~ ., 
                         data = train_data,
                         ntree = 500,
                         mtry = 4,
                         importance = TRUE)
```

**Strengths:**
- Captures non-linear relationships
- Handles interactions automatically
- Provides feature importance
- Robust to outliers

### Train-Test Split
- **Training Set**: 70% of data
- **Test Set**: 30% of data
- **Stratified Sampling**: Maintains class proportions

## Results & Model Comparison

### Logistic Regression Performance

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Overall Accuracy** | 81.9% | Correctly predicts about 4 out of 5 cases |
| **Sensitivity (Recall)** | 0.727 | Detects 72.7% of actual heart attack cases |
| **Specificity** | 0.826 | Correctly identifies 82.6% of non-risk cases |
| **Precision** | 0.219 | About 22% of alerts are true positives |
| **AUC** | 0.864 | Excellent discrimination ability |
| **Balanced Accuracy** | 0.777 | Good performance on both classes |
| **NPV** | 0.980 | Very reliable for negative predictions |

**Confusion Matrix:**
```
              Reference
Prediction    No      Yes
        No    28,816  557
        Yes   6,069   1,883
```

### Random Forest Performance

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Overall Accuracy** | 75.3% | Correctly predicts about 3 out of 4 cases |
| **Sensitivity (Recall)** | 0.808 | Detects 80.8% of actual heart attack cases |
| **Specificity** | 0.750 | Correctly identifies 75% of non-risk cases |
| **Precision** | 0.160 | Only 16% of alerts are true positives |
| **AUC** | 0.870 | Excellent discrimination ability |
| **Balanced Accuracy** | 0.779 | Slightly better than logistic regression |
| **NPV** | 0.985 | Extremely reliable for negative predictions |

**Confusion Matrix:**
```
              Reference
Prediction    No      Yes
        No    33,652  506
        Yes   11,233  2,134
```

### Model Comparison Summary

| Criteria | Logistic Regression | Random Forest |
|----------|---------------------|---------------|
| Overall Accuracy | **81.9%** ⭐ | 75.3% |
| Sensitivity | 72.7% | **80.8%** ⭐ |
| Specificity | **82.6%** ⭐ | 75.0% |
| Precision | **21.9%** ⭐ | 16.0% |
| AUC | 0.864 | **0.870** ⭐ |
| Interpretability | **Excellent** ⭐ | Good |
| Speed | **Very Fast** ⭐ | Moderate |

**Winner**: **Logistic Regression** for overall performance, but **Random Forest** for sensitivity (detecting true positives).

### Feature Importance (Random Forest)

Top predictors ranked by Mean Decrease Accuracy:

1. **HadAngina** (269) - Prior angina diagnosis is strongest predictor
2. **AgeCategory** (109) - Age is second most important
3. **ChestScan** (84) - Prior chest imaging
4. **GeneralHealth** (79) - Self-reported health status
5. **WeightInKilograms** (51) - Weight measurement
6. **BMI** (46) - Body mass index
7. **SmokerStatus** (29) - Smoking history
8. **HadDiabetes** (41) - Diabetes status

## Key Insights

### Model Strengths
✅ **High Sensitivity**: Both models detect most true heart attack cases (73-81%)
✅ **High NPV**: When models predict "no risk", this is reliable 98%+ of the time
✅ **Strong AUC**: Both models show excellent discrimination (0.86-0.87)
✅ **Clinical Relevance**: Models prioritize detecting positive cases (healthcare priority)

### Model Weaknesses
⚠️ **Low Precision**: High false positive rate (79-84% of alerts are false alarms)
⚠️ **Class Imbalance**: Dataset heavily skewed (94.5% negative cases)
⚠️ **Limited Detection Rate**: Only 4-6% of total cases correctly identified
⚠️ **Low Kappa**: Models only slightly better than random prediction (Kappa < 0.20)

### Clinical Implications
- **Screening Tool**: Better used for initial screening to catch high-risk patients
- **Not Diagnostic**: Should complement, not replace, clinical judgment
- **Threshold Tuning**: Different thresholds can optimize sensitivity vs. precision tradeoff
- **False Positive Cost**: Alert fatigue from false positives must be managed

## Project Structure

```
Heart-Attack-Prediction/
├── README.md                                    # This file
├── Heart-attack-detection-project.Rmd          # R Markdown source
├── Heart-attack-dection-project.html           # Rendered HTML report
├── Heart dectection.xlsx                       # Input dataset
└── LICENSE                                      # MIT License
```

## Requirements & Installation

### R Packages
```r
required_packages <- c(
  "readxl",        # Read Excel files
  "dplyr",         # Data manipulation
  "ggplot2",       # Visualization
  "caret",         # Machine learning toolkit
  "randomForest",  # Random Forest modeling
  "pROC"           # ROC curve analysis
)

# Install packages
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}
```

### R Version
- R >= 4.0
- RStudio (recommended)

### System Requirements
- Minimum 4GB RAM
- ~100MB disk space for data and results

## Usage

### Option 1: Run from R/RStudio
```r
# Read the source Rmd file
rmarkdown::render("Heart-attack-detection-project.Rmd")
```

### Option 2: View HTML Report
Open `Heart-attack-dection-project.html` in any web browser

### Option 3: Reproduce Analysis
1. Load the dataset: `data = read_excel('Heart dectection.xlsx')`
2. Follow the sections: Data Cleaning → EDA → Model Building → Evaluation
3. Modify parameters as needed for your use case

## Recommendations & Future Work

### For Immediate Use
1. ✅ Use as **screening tool** for general population
2. ✅ Apply **logistic regression** for interpretability
3. ✅ Set **threshold at 0.3** to maximize sensitivity
4. ⚠️ Always validate with clinical assessment
5. ⚠️ Monitor false positive rates in production

### For Model Improvement
- [ ] **Class Balancing**: Try SMOTE, ROSE, or class weights
- [ ] **Hyperparameter Tuning**: Optimize RF ntree, mtry, depth
- [ ] **Ensemble Methods**: Combine models (voting/stacking)
- [ ] **Advanced Techniques**: XGBoost, Neural Networks
- [ ] **Cost-Sensitive Learning**: Penalize errors based on clinical impact
- [ ] **Cross-Validation**: Use k-fold CV for robust estimates
- [ ] **External Validation**: Test on independent datasets

### For Production Deployment
- [ ] Create prediction API/microservice
- [ ] Implement monitoring for model drift
- [ ] Set up alert thresholds for clinical teams
- [ ] Develop patient-facing risk reports
- [ ] Integrate with EHR/health systems
- [ ] Regular retraining on new data

## Model Assumptions & Limitations

### Assumptions
1. Features are independent (logistic regression)
2. Relationships are approximately linear (logistic regression)
3. Test set distribution matches training set
4. Patient characteristics remain stable over time
5. Data is representative of target population

### Limitations
- **Imbalanced Data**: 5.5% prevalence affects model learning
- **Deterministic**: Doesn't account for temporal changes
- **Cross-sectional**: No longitudinal tracking of outcomes
- **Geographic Bias**: May reflect US population characteristics
- **Recency**: Data currency depends on collection date
- **Generalization**: May not generalize to other populations

## Evaluation Metrics Explained

### Sensitivity (Recall)
- **Definition**: Proportion of actual positives correctly identified
- **Formula**: TP / (TP + FN)
- **Why It Matters**: In healthcare, missing a case is dangerous
- **Trade-off**: Increases false positives

### Specificity
- **Definition**: Proportion of actual negatives correctly identified
- **Formula**: TN / (TN + FP)
- **Why It Matters**: Reduces unnecessary alerts and patient anxiety
- **Trade-off**: May miss some true cases

### Precision
- **Definition**: Proportion of predicted positives that are actually positive
- **Formula**: TP / (TP + FP)
- **Why It Matters**: Indicates reliability of positive predictions
- **Our Value**: 16-22% (many false alarms)

### AUC-ROC
- **Definition**: Area under the Receiver Operating Characteristic curve
- **Range**: 0.5 (random) to 1.0 (perfect)
- **Our Value**: 0.86-0.87 (Excellent)
- **Meaning**: Model distinguishes classes very well

## References & Citations

### Key Papers
- Hart, W. E., et al. (2017). Pyomo - Optimization Modeling in Python. Springer Science+Business Media.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.

### R Documentation
- [randomForest](https://cran.r-project.org/web/packages/randomForest/)
- [ggplot2](https://ggplot2.tidyverse.org/)
- [caret](http://topepo.github.io/caret/)

### Clinical Resources
- American Heart Association (AHA) Risk Calculator
- Framingham Heart Study
- ACC/AHA Cardiovascular Risk Assessment Guidelines

## Contributing

Contributions are welcome! Areas for enhancement:
- [ ] Feature engineering (interaction terms, derived features)
- [ ] Alternative algorithms (SVM, Neural Networks, XGBoost)
- [ ] Visualization improvements
- [ ] Documentation expansion
- [ ] Code optimization

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

## Authors

**Project Team**
- **Dao Van Duc** - Lead Data Scientist
- **Trang Linh To** - Data Analyst
- **Khalil Akchi** - Machine Learning Engineer

**Institution**: Ca' Foscari University of Venice  
**Course**: Advanced Machine Learning & Statistical Modeling  
**Date**: May 15, 2025

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research or work, please cite:

```bibtex
@misc{dao2025heartattack,
  author = {Dao, Van Duc and To, Trang Linh and Akchi, Khalil},
  title = {Heart Attack Prediction: Machine Learning Classification Models},
  year = {2025},
  howpublished = {\url{https://github.com/yourusername/heart-attack-prediction}},
  note = {Course Project - Ca' Foscari University of Venice}
}
```

## Contact & Support

**Questions or Issues?**
- 🐛 Report Bugs: Open an issue on GitHub
- 💡 Feature Requests: Submit a pull request

**Disclaimer**
This model is a research project and should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

---

**Last Updated**: May 15, 2025  
**Version**: 1.0  
**Status**: Complete ✅
