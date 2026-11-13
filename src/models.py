from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_models():
    # Each model is paired with a small hyperparameter grid.
    # The grids are searched with cross-validation on the training split only.
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000),
            {"C": [0.1, 1, 10, 100]}
        ),
        "Linear SVM": (
            LinearSVC(max_iter=20000),
            {"C": [0.1, 1, 10]}
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]}
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [None, 6, 10], "min_samples_leaf": [1, 3, 5]}
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42, n_jobs=4),
            {"n_estimators": [300, 600], "max_features": [1, 2, "sqrt"], "min_samples_leaf": [1, 2]}
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": [200, 400], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
        ),
        "Support Vector Machine": (
            SVC(),
            {"C": [1, 10, 100], "gamma": ["scale", 0.5, 1]}
        ),
        "Multi-layer Perceptron": (
            MLPClassifier(max_iter=3000, random_state=42),
            {"hidden_layer_sizes": [(64,), (64, 32)], "alpha": [1e-4, 1e-2]}
        ),
        "XGBoost": (
            XGBClassifier(random_state=42, eval_metric="mlogloss", n_jobs=4),
            {"n_estimators": [300, 600], "max_depth": [4, 6], "learning_rate": [0.05, 0.1]}
        ),
        "LightGBM": (
            LGBMClassifier(random_state=42, verbosity=-1, n_jobs=4),
            {"n_estimators": [300, 600], "num_leaves": [15, 31], "learning_rate": [0.05, 0.1]}
        ),
    }
