"""TF-IDF classification with cross-validated regularization."""
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def fit_text_classifier(texts, labels, test_size=0.2, seed=42):
    x_train, x_test, y_train, y_test = train_test_split(
        list(texts), list(labels), test_size=test_size, stratify=labels, random_state=seed)
    estimator = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    search = GridSearchCV(estimator, {"classifier__C": [0.1, 1, 10]}, cv=cv,
                          scoring="f1_macro", n_jobs=-1)
    search.fit(x_train, y_train)
    report = classification_report(y_test, search.predict(x_test), output_dict=True, zero_division=0)
    return search.best_estimator_, report, search.best_params_
