import shap
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

class ExplainEngine:

    def __init__(self, model, X_train):

        self.model = model
        self.X_train = X_train

        self.shap_explainer = shap.TreeExplainer(model)

        self.lime_explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns,
            class_names=["Down","Up"],
            mode="classification"
        )

    def shap_explanation(self, sample):
        shap_values = self.shap_explainer(sample)

        values = np.abs(shap_values.values[0])
    
        clean_values = {}
    
        for feature, val in zip(self.X_train.columns, values):
            if isinstance(val, np.ndarray):
                val = float(val[0])
            clean_values[feature] = float(val)
    
        return clean_values
    
    def shap_plot(self, sample):
    
        shap_values = self.shap_explainer(sample)
    
        # SHAP values shape: (features, classes)
        values = shap_values.values[0][:, 0]
    
        feature_names = self.X_train.columns
    
        import pandas as pd
        import matplotlib.pyplot as plt
    
        shap_series = pd.Series(values, index=feature_names)
    
        shap_series.abs().sort_values().plot.barh()
    
        plt.title("Feature Impact on Prediction")
        plt.xlabel("Impact Strength")
        return plt
        
    def lime_explanation(self, sample):

        exp = self.lime_explainer.explain_instance(
            sample.values[0],
            lambda x: self.model.predict_proba(
                pd.DataFrame(x, columns=self.X_train.columns)
            ),
            num_features=5
        )
    
        return exp.as_list()

from fairlearn.metrics import demographic_parity_difference

class ReasonEngine:

    def __init__(self, sensitive_column):

        self.sensitive_column = sensitive_column

    def fairness_check(self, X_test, y_test, y_pred):
        if self.sensitive_column is None:
            return {
                "fairness_check": "No sensitive feature provided"
            }

        sensitive_feature = X_test[self.sensitive_column]

        dp_diff = demographic_parity_difference(
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_feature
        )

        bias_flag = abs(dp_diff) > 0.1

        return {
            "demographic_parity_difference": dp_diff,
            "bias_flag": bias_flag
        }

class RAIWrapper:

    def __init__(self, model, X_train, sensitive_column):

        self.model = model

        self.explain_engine = ExplainEngine(model, X_train)

        self.reason_engine = ReasonEngine(sensitive_column)

    def predict(self, sample):
    
        prediction = self.model.predict(sample)[0]
        confidence = self.model.predict_proba(sample)[0].max()
    
        shap_values = self.explain_engine.shap_explanation(sample)
        lime_rules = self.explain_engine.lime_explanation(sample)
    
        # Convert prediction to human text
        prediction_label = prediction
    
        # Sort features by importance
        ranked_features = sorted(
            shap_values.items(),
            key=lambda x: x[1],
            reverse=True
        )
    
        # Top 3 drivers
        top_drivers = ranked_features[:3]
    
        explanation = []
    
        for feature, value in top_drivers:
            explanation.append(f"{feature} had strong influence ({round(value,3)})")
    
        result = {
        "prediction": prediction_label,
        "prediction_numeric": int(prediction),
        "confidence": float(confidence),
        "shap_values": shap_values,
        "top_factors": top_drivers,
        "lime_rules": lime_rules
        }
        return result

    def fairness_report(self, X_test_full, y_test):

        X_test_model = X_test_full.drop(columns=[self.reason_engine.sensitive_column])
    
        y_pred = self.model.predict(X_test_model)
    
        result = self.reason_engine.fairness_check(
            X_test_full,
            y_test,
            y_pred
        )
    
        dp = float(result["demographic_parity_difference"])
        bias = result["bias_flag"]
    
        report = {
            "Protected Attribute": self.reason_engine.sensitive_column,
            "Prediction Difference (%)": round(dp * 100, 2),
            "Bias Detected": "Yes" if bias else "No significant bias found"
        }
    
        return report

def wrap(model, X_train, sensitive_column=None):
    return RAIWrapper(
        model=model,
        X_train=X_train,
        sensitive_column=sensitive_column
    )
       

# Next

# rai_model = RAIWrapper(
#     model,
#     X_train,
#     sensitive_column="trader_group"
# )
# sample = X_test.iloc[[0]]
# result = rai_model.predict(sample)

# result = rai_model.predict(sample)

# for k, v in result.items():
#     print(f"\n{k}:")
    
#     if isinstance(v, list):
#         for item in v:
#             print("-", item)
#     else:
#         print(v)

# rai_model.explain_engine.shap_plot(sample)


# fairness = rai_model.fairness_report(X_test_full, y_test)

# print(fairness)
