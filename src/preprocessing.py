import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

FEATURES = [
    "LB","AC","FM","UC","DL","DS","DP","ASTV","MSTV","ALTV","MLTV",
    "Width","Min","Max","Nmax","Nzeros","Mode","Mean","Median","Variance","Tendency"
]

N_FEATURES = 5


def remove_duplicates(df):
    # The CTG data contains 13 exact duplicate records.
    # Dropping them keeps copies of a record out of both train and test.
    return df.drop_duplicates().reset_index(drop=True)


def prepare(df, k=N_FEATURES, test_size=0.30, random_state=42):
    df = remove_duplicates(df)
    X = df[FEATURES]
    y = df["NSP"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scaler and selector are fitted on the training split only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    scores = pd.DataFrame({
        "Feature": FEATURES,
        "F Score": selector.scores_,
        "Selected": selector.get_support()
    }).sort_values("F Score", ascending=False)

    return X_train_selected, X_test_selected, y_train, y_test, scores
