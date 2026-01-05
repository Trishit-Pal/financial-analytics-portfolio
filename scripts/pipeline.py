"""
pipeline.py

Run the full analysis pipeline in order:
1. Load data from Alpha Vantage and save cleaned CSV
2. Engineer features and create analysis charts
"""

from scripts import data_loading   # if this errors, see note below
from scripts import feature_analysis

# NOTE:
# If the import syntax above gives errors in your environment, replace with:
#   import scripts.01_data_loading as data_loading
#   import scripts.02_feature_analysis as feature_analysis
# and run `python -m scripts.pipeline` from project root.


def main():
    print("\n=========== RUNNING FULL PIPELINE ===========")

    # Step 1: Fetch + clean
    df_loaded = data_loading.main()

    # Step 2: Feature engineering + analysis
    df_features = feature_analysis.main()

    print("=========== PIPELINE COMPLETE ===========\n")
    return df_features


if __name__ == "__main__":
    main()
