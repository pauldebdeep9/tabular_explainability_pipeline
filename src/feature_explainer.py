import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor

class FeatureExplainer:
    def __init__(self, df, feature_selector, n_train=40, drop_cols=None, max_display=5, sample_no=33):
        self.df = df.dropna(axis=1)
        self.feature_selector = feature_selector
        self.n_train = n_train
        self.drop_cols = drop_cols or []
        self.max_display = max_display
        self.sample_no = sample_no
        self.X, self.y = self._prepare_data()
        self.X_selected = self.feature_selector.select(self.X, self.y)

    def _prepare_data(self):
        df = self.df.drop(columns=self.drop_cols, errors='ignore')
        X = df.drop(['date', 'CP score'], axis=1)
        y = df['CP score']
        return X.iloc[:self.n_train], y.iloc[:self.n_train]

    def explain_global(self):
        from interpret.glassbox import ExplainableBoostingRegressor
        model = ExplainableBoostingRegressor()
        model.fit(self.X_selected, self.y)
        explainer = shap.Explainer(model.predict, shap.utils.sample(self.X_selected, 100))
        shap_values = explainer(self.X_selected)
        shap.summary_plot(shap_values, self.X_selected, max_display=self.max_display, show=False)
        plt.savefig(f"shap_summary_{self.feature_selector.__class__.__name__}.png", dpi=300, bbox_inches="tight")
        return shap_values
    
    def explain_local(self):
        model = GradientBoostingRegressor(random_state=42)
        model.fit(self.X_selected, self.y)

        explainer = shap.Explainer(model.predict, self.X_selected)
        shap_values = explainer(self.X_selected)

        shap.plots.waterfall(shap_values[self.sample_no], max_display=self.max_display)
        
        # Save the plot
        results_dir = Path(__file__).resolve().parent.parent / "Results"
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / f"shap_local_{self.feature_selector.__class__.__name__}.png", dpi=300, bbox_inches='tight')
        plt.close()

        return shap_values[self.sample_no], self.X_selected.iloc[self.sample_no]
