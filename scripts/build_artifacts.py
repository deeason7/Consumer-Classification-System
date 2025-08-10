# scripts/build_artifacts.py
import os
import sys
import pandas as pd
import pickle
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing.transformer import calculate_target_encoding, extreme_negative_keywords, negative_keywords

def build_deployment_artifacts():
    """
    Loads the final processed data to create and save artifacts required for the deployment CLI and Web App.
    This script should be run after the data processing notebooks are complete.
    """
    print("Starting deployment artifact building")

    # Paths
    PROCESSED_DATA_PATH = os.path.join(project_root, "data/processed/consumer_complaints_final.csv")
    OUTPUTS_PATH = os.path.join(project_root, "outputs/")
    os.makedirs(OUTPUTS_PATH, exist_ok=True)

    # Load the final processed data
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: Processed data not found at {PROCESSED_DATA_PATH}")
        print("Please ensure the data cleaning and EDA notebooks (02, 03) have been run successfully.")
        return

    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Loaded processed data with {len(df)} rows.")

    # Calculate and Save Target Encoding Maps for Deployment
    product_map = calculate_target_encoding(df, "product", "sentiment_encoded")
    company_map = calculate_target_encoding(df, "company_grouped", "sentiment_encoded")

    with open(os.path.join(OUTPUTS_PATH, "product_dispute_map.pkl"), "wb") as f:
        pickle.dump(product_map, f)
    print("Saved product dispute map.")

    with open(os.path.join(OUTPUTS_PATH, "company_dispute_map.pkl"), "wb") as f:
        pickle.dump(company_map, f)
    print("Saved company dispute map.")

    # Extract full lists of products and companies for UI dropdowns
    all_products = sorted(df['product'].unique().tolist())
    all_companies = sorted(df['company_grouped'].unique().tolist())

    print(f"Extracted {len(all_products)} unique products and {len(all_companies)} unique companies.")

    #  Save Feature Configuration for Deployment
    feature_config = {
        "global_mean_dispute_rate": df['sentiment_encoded'].mean(),
        "extreme_negative_keywords": extreme_negative_keywords,
        "negative_keywords": negative_keywords,
        "all_products": all_products,
        "all_companies": all_companies
    }

    with open(os.path.join(OUTPUTS_PATH, "feature_config.json"), "w") as f:
        json.dump(feature_config, f, indent=2)
    print("Saved feature configuration file.")

    print("\nDeployment artifacts built successfully!")

if __name__ == "__main__":
    build_deployment_artifacts()