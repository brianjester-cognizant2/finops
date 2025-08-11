import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

DATA_DIR = "data"
BILLING_DATA_FILE = os.path.join(DATA_DIR, "billing_data.parquet")

def get_billing_data(start_date, end_date, project_filter=None):
    """
    Mocks a call to a billing API, returning detailed usage and cost data
    for GenAI services, structured similarly to a Google Cloud Billing export.
    
    Args:
        start_date (datetime.date): The start of the date range.
        end_date (datetime.date): The end of the date range.
        project_filter (list, optional): A list of project names to filter by. 
                                         If None, data for all projects is returned.
    
    Returns:
        dict: A dictionary containing billing information.
    """
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    services = {
        "OpenAI GPT API": {
            "models": {
                "GPT-4.1 (Base Model)": {"input": 2.00 / 1_000_000, "output": 8.00 / 1_000_000},
                "GPT-4.1-mini":         {"input": 0.40 / 1_000_000, "output": 1.60 / 1_000_000},
                "GPT-4.1-nano":         {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
                "GPT-4":                {"input": 30.00 / 1_000_000, "output": 60.00 / 1_000_000},
                "GPT-4 Turbo":          {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
                "GPT-4o":               {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
                "GPT-4o Mini":          {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
            }
        }
    }
    
    all_projects = ['GenAI-Chat', 'ML-Pipeline', 'Analytics-Engine', 'Cognitive-Services', 'MLOps-Platform', 'Data-Lake', 'Vision-API', 'Speech-Processing', 'Bot-Framework', 'Document-AI', 'Data-Warehouse', 'BI-Analytics', 'ML-Features', 'Real-time-Analytics', 'Feature-Store']
    all_projects_set = set(all_projects)
    
    # If a filter is provided, use it. Otherwise, use all projects.
    if project_filter:
        # Ensure we only process projects that exist in our mock data
        projects_to_process = [p for p in project_filter if p in all_projects_set]
    else:
        projects_to_process = all_projects

    billing_records = []
    
    for date in dates:
        for project_name in projects_to_process:
            for service_name, service_details in services.items():
                for model_name, prices in service_details["models"].items():
                    # Reduce record generation by 75%. The original generation rate was 70% (1 - 0.3).
                    # The new rate is 17.5% (0.7 * 0.25), achieving a 75% reduction.
                    if np.random.random() > 0.02:
                        continue

                    # --- Generate Input Token Record ---
                    input_usage_amount = np.random.randint(500_000, 20_000_000)
                    input_cost = input_usage_amount * prices["input"] * np.random.uniform(0.95, 1.05)
                    input_sku_desc = f"{model_name} - Input Tokens"
                    
                    input_record = {
                        "service": {"id": f"services/{service_name.lower().replace(' ', '-')}", "description": service_name},
                        "sku": {"id": f"skus/{input_sku_desc.lower().replace(' ', '-')}", "description": input_sku_desc},
                        "usage_start_time": date.isoformat() + "Z",
                        "usage_end_time": (date + timedelta(days=1) - timedelta(seconds=1)).isoformat() + "Z",
                        "project": {"id": f"projects/{project_name.lower()}", "name": project_name},
                        "cost": round(input_cost, 2),
                        "currency": "USD",
                        "usage": {"amount": input_usage_amount, "unit": "tokens"}
                    }
                    billing_records.append(input_record)

                    # --- Generate Output Token Record ---
                    output_usage_amount = int(input_usage_amount * np.random.uniform(0.1, 0.5))
                    output_cost = output_usage_amount * prices["output"] * np.random.uniform(0.95, 1.05)
                    output_sku_desc = f"{model_name} - Output Tokens"

                    output_record = {
                        "service": {"id": f"services/{service_name.lower().replace(' ', '-')}", "description": service_name},
                        "sku": {"id": f"skus/{output_sku_desc.lower().replace(' ', '-')}", "description": output_sku_desc},
                        "usage_start_time": date.isoformat() + "Z",
                        "usage_end_time": (date + timedelta(days=1) - timedelta(seconds=1)).isoformat() + "Z",
                        "project": {"id": f"projects/{project_name.lower()}", "name": project_name},
                        "cost": round(output_cost, 2),
                        "currency": "USD",
                        "usage": {"amount": output_usage_amount, "unit": "tokens"}
                    }
                    billing_records.append(output_record)
                    
    return {
        "billingInfo": billing_records,
        "nextPageToken": None # Mocking a single page response for simplicity
    }

def _generate_and_save_billing_data():
    """Generates a full set of billing data for a standard period and saves it."""
    print("Generating a full set of mock billing data...")
    # Use a fixed, wide date range for pre-generation
    start_date_gen = datetime(2023, 1, 1).date()
    end_date_gen = datetime.today().date()
    
    billing_data_response = get_billing_data(start_date_gen, end_date_gen, project_filter=None)
    billing_records = billing_data_response.get("billingInfo", [])

    if not billing_records:
        print("Warning: No billing records were generated.")
        return pd.DataFrame()

    # Flatten the nested dictionary structure
    flat_records = [
        {
            "Date": pd.to_datetime(rec["usage_start_time"]).date(),
            "Project": rec["project"]["name"],
            "Service": rec["service"]["description"],
            "SKU": rec["sku"]["description"],
            "Cost (USD)": rec["cost"],
            "Usage": rec["usage"]["amount"],
            "Unit": rec["usage"]["unit"]
        }
        for rec in billing_records
    ]
    billing_df = pd.DataFrame(flat_records)
    
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        billing_df.to_parquet(BILLING_DATA_FILE, index=False)
        print(f"✅ Billing data saved to {BILLING_DATA_FILE}")
        return billing_df
    except Exception as e:
        print(f"❌ Error saving billing data: {e}")
        return None

def load_or_generate_billing_data(force_regenerate=False):
    """Loads billing data from file, or generates it if it's missing or regeneration is forced."""
    if not force_regenerate and os.path.exists(BILLING_DATA_FILE) and os.path.getsize(BILLING_DATA_FILE) > 0:
        print("Loading billing data from file...")
        return pd.read_parquet(BILLING_DATA_FILE)
    
    return _generate_and_save_billing_data()