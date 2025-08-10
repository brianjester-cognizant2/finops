import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
        "Generative Language API": {
            "skus": {
                "GPT-4 Tokens": 0.06 / 1000, # Cost per token
                "Claude 3 Tokens": 0.025 / 1000,
                "Llama 3 Tokens": 0.002 / 1000,
                "Mixtral Tokens": 0.001 / 1000,
                "PaLM 2 Tokens": 0.002 / 1000
            }
        },
        "Vertex AI Conversation": {
            "skus": {
                "Chat Session": 0.005 # Cost per session
            }
        }
    }
    
    all_projects = ['GenAI-Chat', 'ML-Pipeline', 'Analytics-Engine', 'Cognitive-Services', 'MLOps-Platform', 'Data-Lake', 'Vision-API', 'Speech-Processing', 'Bot-Framework', 'Document-AI', 'Data-Warehouse', 'BI-Analytics', 'ML-Features', 'Real-time-Analytics', 'Feature-Store']
    
    # If a filter is provided, use it. Otherwise, use all projects.
    projects_to_process = project_filter if project_filter else all_projects

    billing_records = []
    
    for date in dates:
        for project_name in projects_to_process:
            if project_name not in all_projects:
                continue

            for service_name, service_details in services.items():
                for sku_name, base_cost in service_details["skus"].items():
                    if np.random.random() > 0.6:
                        continue

                    if "Tokens" in sku_name:
                        usage_amount = np.random.randint(1_000_000, 50_000_000)
                        usage_unit = "tokens"
                        cost = usage_amount * base_cost * np.random.uniform(0.9, 1.1)
                    elif "Session" in sku_name:
                        usage_amount = np.random.randint(100, 2000)
                        usage_unit = "sessions"
                        cost = usage_amount * base_cost * np.random.uniform(0.9, 1.1)
                    
                    record = {
                        "service": {"id": f"services/{service_name.lower().replace(' ', '-')}", "description": service_name},
                        "sku": {"id": f"skus/{sku_name.lower().replace(' ', '-')}", "description": sku_name},
                        "usage_start_time": date.isoformat() + "Z",
                        "usage_end_time": (date + timedelta(days=1) - timedelta(seconds=1)).isoformat() + "Z",
                        "project": {"id": f"projects/{project_name.lower()}", "name": project_name},
                        "cost": round(cost, 2),
                        "currency": "USD",
                        "usage": {"amount": usage_amount, "unit": usage_unit}
                    }
                    billing_records.append(record)
                    
    return {
        "billingInfo": billing_records,
        "nextPageToken": None # Mocking a single page response for simplicity
    }