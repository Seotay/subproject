
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('./data/filtered_data/filtered_train.tsv', sep='\t')
test_df = pd.read_csv('./data/filtered_data/filtered_test.tsv', sep='\t')
print('shpae of train data: ', train_df.shape)
print('shpae of test data: ', test_df.shape)


train_df.head()


feature_cols = [col for col in train_df.columns if col not in ["총전기사용량"]]

X = train_df[feature_cols].copy()
y = train_df["총전기사용량"].values.copy()
y = y.reshape(-1, 1)

# Target Label Encoding
from sklearn.preprocessing import RobustScaler

# 1. 스케일러 생성

minmax_target = RobustScaler()
scaled_y = minmax_target.fit_transform(y)

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
X_test = test_df.copy()
encoders = {}  # 각 컬럼별 encoder 저장

for col in categorical_features:
    le_train = LabelEncoder()
    X[col] = le_train.fit_transform(X[col])
    encoders[col] = le_train
    unseen_labels_val = set(X_test[col]) - set(le_train.classes_)
    if unseen_labels_val:
        le_train.classes_ = np.append(le_train.classes_, list(unseen_labels_val))
    X_test[col] = le_train.transform(X_test[col])
    
print('Shape of train X', X.shape)
print('Shape of train y', scaled_y.shape)
print('Shape of test X', X_test.shape) # original index 포함되어있음.



kf = KFold(n_splits=5, shuffle=True, random_state=42)
models = [] 


rmse_scores = []
mse_scores = []
mae_scores = []
r2_scores = []


for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = scaled_y[train_idx], scaled_y[val_idx]

    print('-'*40)
    print(f'Fold {fold + 1}-th XGBoost model training...')
    
    # XGBoost
    #model = xgb.XGBRegressor(
    #    tree_method='gpu_hist', 
    #    gpu_id=0,
    #    random_state=42
    #)
    model = xgb.XGBRegressor(n_estimators=100)

    # training and validation mornitoring
    model.fit(X_train, y_train)
    models.append(model)
    
    y_val_pred = model.predict(X_val)
    
    y_val_pred = minmax_target.inverse_transform(y_val_pred.reshape(-1, 1))
    y_val = minmax_target.inverse_transform(y_val.reshape(-1, 1))
    
    mae = mean_absolute_error(y_val, y_val_pred)
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_val_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print('-'*40)
    
    rmse_scores.append(rmse)
    mse_scores.append(mse)
    mae_scores.append(mae)
    r2_scores.append(r2)
    
    
print(f"K-Fold mean RMSE: {np.mean(rmse_scores):.4f}")
print(f"K-Fold mean MSE: {np.mean(mse_scores):.4f}")
print(f"K-Fold mean MAE: {np.mean(mae_scores):.4f}")
print(f"K-Fold mean R2: {np.mean(r2_scores):.4f}")



original_index = X_test['index_origin']
X_test = X_test.drop(columns=['index_origin'])

y_preds = np.zeros((len(X_test), len(models)))

for i, model in enumerate(models):    
    predicted_value = model.predict(X_test)
    inverse_predicted_value = minmax_target.inverse_transform(predicted_value.reshape(-1, 1))
    y_preds[:, i] = inverse_predicted_value.flatten()

y_preds = y_preds.mean(axis=1)
mean_preds_df = pd.DataFrame(y_preds, columns=["y_pred"])
mean_preds_df


mean_preds_df.to_csv('./results/baseline_submission(k5fold-xgboost).csv', sep=',', index=False)


