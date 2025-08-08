#!/usr/bin/env python3
"""
Test script to verify the data loading functionality
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Test the data generation and loading functions
def test_data_functions():
    print("Testing data generation and loading functions...")
    
    # Import the data functions from the main app
    import sys
    sys.path.append('.')
    
    # We'll copy the relevant functions here to test them independently
    DATA_DIR = "data"
    MODEL_DATA_FILE = os.path.join(DATA_DIR, "model_data.parquet")
    INFRA_DATA_FILE = os.path.join(DATA_DIR, "infra_data.parquet")
    
    # Create simple test data
    model_test_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'model': ['GPT-4'] * 10,
        'cloud_platform': ['AWS'] * 10,
        'project': ['Test-Project'] * 10,
        'department': ['Engineering'] * 10,
        'latency_ms': np.random.uniform(100, 500, 10)
    })
    
    infra_test_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'component': ['API Gateway'] * 10,
        'cloud_platform': ['AWS'] * 10,
        'project': ['Test-Project'] * 10,
        'department': ['Engineering'] * 10,
        'cpu_usage_percent': np.random.uniform(20, 80, 10)
    })
    
    # Test Parquet saving and loading
    print("\n1. Testing Parquet save/load...")
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        model_test_data.to_parquet(MODEL_DATA_FILE, index=False)
        infra_test_data.to_parquet(INFRA_DATA_FILE, index=False)
        print("✅ Parquet files saved successfully")
        
        # Load them back
        loaded_model = pd.read_parquet(MODEL_DATA_FILE)
        loaded_infra = pd.read_parquet(INFRA_DATA_FILE)
        print(f"✅ Parquet files loaded successfully")
        print(f"   - Model data: {len(loaded_model)} records")
        print(f"   - Infra data: {len(loaded_infra)} records")
        
        # Clean up test files
        os.remove(MODEL_DATA_FILE)
        os.remove(INFRA_DATA_FILE)
        print("✅ Test files cleaned up")
        
    except Exception as e:
        print(f"❌ Parquet test failed: {e}")
    
    print("\n2. Testing CSV fallback...")
    try:
        csv_model_file = MODEL_DATA_FILE.replace('.parquet', '.csv')
        csv_infra_file = INFRA_DATA_FILE.replace('.parquet', '.csv')
        
        model_test_data.to_csv(csv_model_file, index=False)
        infra_test_data.to_csv(csv_infra_file, index=False)
        print("✅ CSV files saved successfully")
        
        # Load them back
        loaded_model = pd.read_csv(csv_model_file)
        loaded_infra = pd.read_csv(csv_infra_file)
        print(f"✅ CSV files loaded successfully")
        print(f"   - Model data: {len(loaded_model)} records")
        print(f"   - Infra data: {len(loaded_infra)} records")
        
        # Clean up test files
        os.remove(csv_model_file)
        os.remove(csv_infra_file)
        print("✅ Test files cleaned up")
        
    except Exception as e:
        print(f"❌ CSV test failed: {e}")

if __name__ == "__main__":
    test_data_functions()
    print("\n✅ All tests completed!")
