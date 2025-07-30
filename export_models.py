import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, SGDRegressor, HuberRegressor, TheilSenRegressor, RANSACRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
import joblib

# Load data
file_path = 'Data/Student_Performance.csv'
df = pd.read_csv(file_path)
df.columns = ['hours_studied', 'previous_score', 'extra_curr_activities', 'sleep_hours', 'simple_ques_papers_prac', 'performance_index']
df = df.drop_duplicates()

X = df.drop(columns=['performance_index'])
y = df['performance_index']

numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Save preprocessor
joblib.dump(preprocessor, 'preprocessor.joblib')

# Models to train
models = [
    ("LinearRegression", LinearRegression()),
    ("Ridge", Ridge(random_state=42)),
    ("Lasso", Lasso(random_state=42)),
    ("ElasticNet", ElasticNet(random_state=42)),
    ("BayesianRidge", BayesianRidge()),
    ("SGDRegressor", SGDRegressor(random_state=42, max_iter=1000, tol=1e-3)),
    ("HuberRegressor", HuberRegressor(max_iter=1000)),
    ("TheilSenRegressor", TheilSenRegressor(random_state=42)),
    ("RANSACRegressor", RANSACRegressor(random_state=42, min_samples=0.5)),
    ("DecisionTreeRegressor", DecisionTreeRegressor(random_state=42)),
    ("RandomForestRegressor", RandomForestRegressor(random_state=42, n_estimators=100)),
    ("GradientBoostingRegressor", GradientBoostingRegressor(random_state=42, n_estimators=100)),
    ("AdaBoostRegressor", AdaBoostRegressor(random_state=42, n_estimators=100)),
    ("KNeighborsRegressor", KNeighborsRegressor()),
    ("SVR_rbf", SVR(kernel='rbf')),
    ("SVR_linear", SVR(kernel='linear')),
    ("MLPRegressor", MLPRegressor(random_state=42, max_iter=500, early_stopping=True, n_iter_no_change=50)),
    ("PLSRegression", PLSRegression(n_components=2)),
    ("KernelRidge", KernelRidge(alpha=1.0, kernel='rbf'))
]

for name, model in models:
    model.fit(X_train_transformed, y_train)
    joblib.dump(model, f'{name}.joblib')
print('All models and preprocessor saved.')
