import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

try:
    import scikitplot as skplt
    print("--> [✓] Scikit-plot imported successfully.")
except ImportError as e:
    print(f"--> [X] Import Error: {e}")
    sys.exit(1)

def test_plots():
    print("--- Scikit-Plot Archaeology (#32) ---")
    try:
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(X_train, y_train)
        y_probas = clf.predict_proba(X_test)

        # This call relies on internal sklearn.metrics structures that no longer exist
        print("--> Generating ROC Curve...")
        skplt.metrics.plot_roc(y_test, y_probas)
        
        print("    [✓] Success! Plot generated.")
    except Exception as e:
        print(f"--> [!] MODERNIZATION FAILURE: {type(e).__name__}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_plots()