
# src/run_pipeline.py

from pathlib import Path
import pandas as pd
from src import LassoFeatureSelector, FeatureExplainer, RFECVFeatureSelector

def main():
    # --- Configuration ---
    data_file = Path(__file__).resolve().parent / "Data" / "CPExplain221014.csv"
    drop_cols = ['Man_pmi_mx', 'Man_pmi_tr']
    n_train = 40
    max_display = 5
    sample_no = 33

    # --- Load Data ---
    df = pd.read_csv(data_file)
    print(f"✅ Loaded data shape: {df.shape}")

    # --- Feature Selector ---
    # selector = LassoFeatureSelector(cv=5, random_state=42, max_iter=10000)

    selector = RFECVFeatureSelector(step=0.05, cv=3, scoring="neg_mean_absolute_error", random_state=101)

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
