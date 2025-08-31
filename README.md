# 🧠 Prometheus Insights - GenAI FinOps Dashboard

A professional, enterprise-ready analytics platform for monitoring and analyzing GenAI infrastructure performance, costs, and operational metrics across multiple cloud platforms.

## Features

- **Multi-Cloud Support**: AWS, GCP, Azure, Snowflake, Databricks
- **Department-wise Analytics**: Engineering, Data Science, Marketing, Finance, Operations
- **Real-time Metrics**: Model performance, infrastructure monitoring, cost analysis
- **Interactive Dashboards**: Drill-down capabilities with dynamic filtering
- **Performance Optimized**: Fast data loading with persistent storage

## Prerequisites

### On a home computer

1. Install the Python Install Manager from the Microsoft Store

2. Switch to Python 3.9
```
py install 3.9
py -3.9
```

## Installation

### Cognizant laptop

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run genai_architecture_explorer.py
```

### Home Computer

1. Install dependencies:
```
py -m pip install -r requirements.txt
```

2. Run the application:
```
python -m streamlit run genai_architecture_explorer.py
```

## Data Management

### Performance Optimization
The application now uses **persistent data storage** for improved performance:

- **Parquet Files**: Primary storage format for fast loading and smaller file sizes
- **CSV Fallback**: Automatic fallback if PyArrow is not available
- **Static Data**: Pre-generated data files eliminate slow dynamic generation on every launch

### Data Storage Location
- Data files are stored in the `data/` directory
- Model data: `data/model_data.parquet` (or `.csv`)
- Infrastructure data: `data/infra_data.parquet` (or `.csv`)

### Data Management Controls
The sidebar includes data management controls:

- **📁 Data Status**: Shows when data was last updated
- **🔄 Regenerate Data**: Generate new sample data (useful for testing/development)
- **🗑️ Clear Data Files**: Delete saved files to force regeneration on next load

### First-Time Setup
On first launch:
1. The app generates sample data (this may take a moment)
2. Data is automatically saved to files for future use
3. Subsequent launches load data from files instantly

### Performance Benefits
- **Faster Loading**: Data loads from files instead of generating dynamically
- **Consistent Data**: Same dataset across sessions until manually regenerated
- **Reduced CPU Usage**: No computation overhead on app startup
- **Better User Experience**: Near-instant dashboard loading

## Architecture

### Project-Department Mapping
Each project belongs to exactly one department:
- **AWS Projects**: GenAI-Chat, ML-Pipeline, Data-Lake
- **GCP Projects**: Analytics-Engine, Vision-API, Speech-Processing
- **Azure Projects**: Cognitive-Services, Bot-Framework, Document-AI
- **Snowflake Projects**: Data-Warehouse, BI-Analytics, ML-Features
- **Databricks Projects**: MLOps-Platform, Real-time-Analytics, Feature-Store

### Data Models
- **Model Performance**: Latency, throughput, accuracy, cost metrics
- **Infrastructure Monitoring**: CPU, memory, requests, errors, response times
- **Token Management**: Usage tracking, limits, exceptions

## Troubleshooting

### PyArrow Installation Issues
If you encounter issues with Parquet files:
1. Install PyArrow: `pip install pyarrow`
2. Or let the app automatically fall back to CSV format

### Data Generation Issues
If data seems outdated or incorrect:
1. Use the "🔄 Regenerate Data" button in the sidebar
2. Or manually delete files in the `data/` directory

### Performance Issues
For optimal performance:
1. Ensure PyArrow is installed for Parquet support
2. Keep data files in the `data/` directory
3. Use the regenerate option sparingly (only when needed)

## Development

### File Structure
```
finops/
├── genai_architecture_explorer.py    # Main application
├── requirements.txt                   # Dependencies
├── test_data_loading.py              # Data loading tests
├── data/                             # Data storage directory
│   ├── model_data.parquet           # Model performance data
│   └── infra_data.parquet           # Infrastructure data
└── assets/
    └── logo.png                     # Application logo
```

### Adding New Features
When modifying data generation:
1. Update the `generate_sample_data()` function
2. Test with `python test_data_loading.py`
3. Use "🔄 Regenerate Data" to refresh the dashboard

### Data Format Changes
If you modify the data structure:
1. Update both model and infrastructure data generation
2. Clear existing data files to force regeneration
3. Ensure backward compatibility or update the data version

## Support

For issues or questions:
1. Check the sidebar data management controls
2. Verify all dependencies are installed
3. Run the test script: `python test_data_loading.py`
