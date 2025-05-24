# src/run_pipeline.py
import argparse
from pathlib import Path
import pandas as pd
from src import LassoFeatureSelector, FeatureExplainer, RFECVFeatureSelector

def main():
    # --- Argument parser ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_selection_method", type=str, default="lasso", choices=["lasso", "rfecv"],
                        help="Feature selection method to use: 'lasso' or 'rfecv'")
    args = parser.parse_args()

    # --- Configuration ---
    data_file = Path(__file__).resolve().parent / "Data" / "CPExplain221014.csv"
    drop_cols = ['Man_pmi_mx', 'Man_pmi_tr']
    n_train = 40
    max_display = 6
    sample_no = 33

    # --- Load Data ---
    df = pd.read_csv(data_file)
    print(f"✅ Loaded data shape: {df.shape}")

    # --- Select feature selection strategy ---
    if args.feature_selection_method == "lasso":
        selector = LassoFeatureSelector(cv=5, random_state=42, max_iter=10000)
    elif args.feature_selection_method == "rfecv":
        selector = RFECVFeatureSelector(step=0.05, cv=5, scoring="neg_mean_absolute_error", random_state=101)
    else:
        raise ValueError("Invalid feature selection method. Choose 'lasso' or 'rfecv'.")

    # --- Explanation Pipeline ---
    explainer = FeatureExplainer(
        df=df,
        feature_selector=selector,
        n_train=n_train,
        drop_cols=drop_cols,
        max_display=max_display,
        sample_no=sample_no
    )

    # --- Global Explanation ---
    print("🔍 Generating global SHAP explanation...")
    shap_values = explainer.explain_global()
    print("✅ Global explanation completed and saved.")

    # --- Local Explanation ---
    print("🔍 Generating local SHAP explanation...")
    local_shap_val, instance = explainer.explain_local()
    print(f"✅ Local explanation for instance {sample_no} completed and saved.")

if __name__ == "__main__":
    main()
