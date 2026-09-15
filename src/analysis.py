"""Reproducible customer-behaviour classification analysis.

Run from the repository root: ``python src/analysis.py``.
The script creates metrics, accessible figures, and a PDF report from one run.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, average_precision_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
FIGURES, RESULTS, RANDOM_STATE, CV_MAX_ROWS = ROOT / "figures", ROOT / "results", 42, 1_000

def preprocessor(X):
    categorical = X.select_dtypes(include=["str", "object", "bool", "category"]).columns.tolist()
    numeric = X.columns.difference(categorical).tolist()
    return ColumnTransformer([("numeric", StandardScaler(), numeric), ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)], verbose_feature_names_out=False)

def candidate_models(transformer):
    def pipe(model): return Pipeline([("preprocess", transformer), ("model", model)])
    return {
        "Logistic regression": pipe(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
        "Linear SVM": pipe(SVC(kernel="linear", class_weight="balanced")),
        "Polynomial SVM": pipe(SVC(kernel="poly", degree=3, class_weight="balanced")),
        "RBF SVM": pipe(SVC(kernel="rbf", class_weight="balanced")),
    }

def evaluate(model, name, X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    if len(X_train) > CV_MAX_ROWS:
        X_cv, _, y_cv, _ = train_test_split(X_train, y_train, train_size=CV_MAX_ROWS, stratify=y_train, random_state=RANDOM_STATE)
    else:
        X_cv, y_cv = X_train, y_train
    scores = cross_validate(model, X_cv, y_cv, cv=cv, n_jobs=1, scoring={"f1": "f1", "ap": "average_precision", "roc": "roc_auc"})
    fitted = model.fit(X_train, y_train)
    predicted, decision = fitted.predict(X_test), fitted.decision_function(X_test)
    return {"model": name, "cv_f1_positive_mean": float(scores["test_f1"].mean()), "cv_f1_positive_sd": float(scores["test_f1"].std()), "cv_average_precision_mean": float(scores["test_ap"].mean()), "test_f1_positive": float(f1_score(y_test, predicted, zero_division=0)), "test_average_precision": float(average_precision_score(y_test, decision)), "test_roc_auc": float(roc_auc_score(y_test, decision)), "classification_report": classification_report(y_test, predicted, output_dict=True, zero_division=0)}, fitted

def decision_slice(model, X_train, X_test, y_test, x_name, y_name, title, output):
    """A valid 2-D slice: non-plotted fields remain at training medians/modes."""
    reference = {c: (X_train[c].median() if pd.api.types.is_numeric_dtype(X_train[c]) else X_train[c].mode().iat[0]) for c in X_train}
    xv, yv = np.linspace(X_train[x_name].quantile(.01), X_train[x_name].quantile(.99), 180), np.linspace(X_train[y_name].quantile(.01), X_train[y_name].quantile(.99), 180)
    xx, yy = np.meshgrid(xv, yv)
    grid = pd.DataFrame([reference] * xx.size); grid[x_name], grid[y_name] = xx.ravel(), yy.ravel()
    score = model.decision_function(grid).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    filled = ax.contourf(xx, yy, score, levels=17, cmap="RdBu_r", alpha=.72)
    ax.contour(xx, yy, score, levels=[0], colors="#202124", linewidths=2)
    for value, label, marker, color in [(0, "Class 0", "o", "#2b6cb0"), (1, "Class 1", "^", "#c53030")]:
        mask = y_test.to_numpy() == value
        ax.scatter(X_test.loc[mask, x_name], X_test.loc[mask, y_name], label=label, marker=marker, color=color, alpha=.42, s=20, edgecolors="none")
    fig.colorbar(filled, ax=ax, label="Model decision score")
    ax.set(title=title, xlabel=x_name, ylabel=y_name); ax.legend(title="Observed target")
    ax.text(.01, .01, "Other features fixed at training median/mode; black line = threshold.", transform=ax.transAxes, fontsize=8, bbox={"facecolor":"white", "alpha":.85, "edgecolor":"none"})
    fig.savefig(output, dpi=180, bbox_inches="tight"); plt.close(fig)

def comparison(metrics, dataset, output):
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True); x, width = np.arange(len(metrics)), .36
    ax.bar(x-width/2, metrics.test_f1_positive, width, label="Test F1 (positive class)", color="#2463a5")
    ax.bar(x+width/2, metrics.test_average_precision, width, label="Test average precision", color="#d97706")
    ax.set(xticks=x, xticklabels=metrics.model, ylim=(0, 1), ylabel="Score", title=f"{dataset}: held-out test performance"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.savefig(output, dpi=180, bbox_inches="tight"); plt.close(fig)

def confusion(model, X_test, y_test, title, output):
    fig, ax = plt.subplots(figsize=(5.8,4.8), constrained_layout=True)
    ConfusionMatrixDisplay.from_predictions(y_test, model.predict(X_test), display_labels=["Class 0", "Class 1"], cmap="Blues", ax=ax, colorbar=False); ax.set_title(title)
    fig.savefig(output, dpi=180, bbox_inches="tight"); plt.close(fig)

def analyze(name, csv, target, drop, slice_features, categorical_codes=()):
    data = pd.read_csv(ROOT / csv); X, y = data.drop(columns=[target, *drop]), data[target].astype(int)
    # Integer IDs below are nominal categories, not ordered measurements.
    for column in categorical_codes:
        X[column] = X[column].astype("str")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, stratify=y, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for model_name, model in candidate_models(preprocessor(X)).items():
        row, fitted[model_name] = evaluate(model, model_name, X_train, y_train, X_test, y_test); rows.append(row)
    metrics = pd.DataFrame(rows); best = metrics.sort_values("cv_f1_positive_mean", ascending=False).iloc[0].model; slug = name.lower().replace(" ", "_")
    comparison(metrics, name, FIGURES/f"{slug}_model_comparison.png")
    decision_slice(fitted[best], X_train, X_test, y_test, *slice_features, f"{name}: {best} decision slice", FIGURES/f"{slug}_decision_slice.png")
    confusion(fitted[best], X_test, y_test, f"{name}: {best} held-out test", FIGURES/f"{slug}_confusion_matrix.png")
    metrics.drop(columns="classification_report").to_csv(RESULTS/f"{slug}_metrics.csv", index=False)
    return {"name":name, "rows":len(data), "features":X.shape[1], "duplicates":int(data.duplicated().sum()), "positive_rate":float(y.mean()), "best_model":best, "metrics":rows}

def report(analyses):
    with PdfPages(ROOT / "Project 7 - Logistic Regression and SVMs.pdf") as pdf:
        fig = plt.figure(figsize=(8.5,11)); fig.text(.09,.94,"Customer Behavior Analysis",fontsize=24,fontweight="bold",color="#17365d"); fig.text(.09,.90,"Logistic Regression and Support Vector Machines",fontsize=13,color="#4a5568")
        fig.text(.09,.82,"Reproducible analysis: preprocessing is fit only on stratified training data; three-fold cross-validation on a stratified training sample compares models; final results are measured once on an untouched held-out test set.",fontsize=11,wrap=True,va="top")
        y=.70
        for item in analyses:
            best=next(m for m in item["metrics"] if m["model"]==item["best_model"])
            fig.text(.09,y,f"{item['name']}\n{item['rows']:,} rows | {item['features']} predictors | positive class {item['positive_rate']:.1%} | duplicate rows {item['duplicates']}\nBest CV model: {item['best_model']} (positive F1 {best['cv_f1_positive_mean']:.3f} +/- {best['cv_f1_positive_sd']:.3f}).\nHeld-out test: F1 {best['test_f1_positive']:.3f}, average precision {best['test_average_precision']:.3f}, ROC-AUC {best['test_roc_auc']:.3f}.",fontsize=11,va="top",linespacing=1.45); y-=.22
        fig.text(.09,.17,"Interpretation",fontsize=14,fontweight="bold",color="#17365d"); fig.text(.09,.13,"Balanced class weights are used as a reproducible baseline. Charts use a valid two-dimensional decision slice: every unplotted feature is fixed to a representative training value, avoiding fabricated customer profiles.",fontsize=10,wrap=True,va="top"); fig.text(.09,.06,"Sources: dataset links are listed in README.",fontsize=8,color="#4a5568")
        pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
        for item in analyses:
            slug=item["name"].lower().replace(" ","_"); fig,axes=plt.subplots(3,1,figsize=(8.5,11),constrained_layout=True); fig.suptitle(f"{item['name']}: validated results",fontsize=18,fontweight="bold",color="#17365d")
            for ax,suffix in zip(axes,["model_comparison","decision_slice","confusion_matrix"]): ax.imshow(plt.imread(FIGURES/f"{slug}_{suffix}.png")); ax.axis("off")
            pdf.savefig(fig); plt.close(fig)

def run_analysis():
    FIGURES.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    analyses=[analyze("Online shoppers","online_shoppers_intention.csv","Revenue",[],("PageValues","ExitRates"), ("OperatingSystems","Browser","Region","TrafficType")), analyze("Bank churn","Bank Customer Churn Prediction.csv","churn",["customer_id"],("balance","age"))]
    (RESULTS/"analysis_summary.json").write_text(json.dumps(analyses,indent=2),encoding="utf-8"); report(analyses); return analyses
if __name__ == "__main__":
    for item in run_analysis(): print(f"{item['name']}: best model is {item['best_model']}")
