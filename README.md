# ⚡️ Electricity Usage Prediction

This project aims to predict electricity usage for different commercial districts based on data provided by the R Institute.

## 📂 Dataset
- **TRAIN_DATA.csv**: Training data for model development.  
- **TEST_DATA.csv**: Evaluation data where predicted values (`y_pred`) should be saved for submission.

## 🎯 Goal: RMSE (Root Mean Square Error)

$$ RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 } $$
- $$y_i$$ = Actual electricity usage  
- $$ŷ_i$$ = Predicted electricity usage  
- **n** = Number of observations  

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt




