import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import os
import mock_billing_service

# Set page configuration
st.set_page_config(
    page_title="Prometheus Insights", # Removed emoji
    page_icon="assets/logo.png", # Use logo for page icon
    layout="wide"
)

# Add custom CSS for metric cards
st.markdown("""
<style>
    .metric-container {
        padding: 1rem;
        border-radius: 0;
        text-align: left;
        border: none;
        margin-bottom: 1rem;
    }
    .metric-container .metric-title {
        font-size: 1rem;
        font-weight: 500;
        color: #000000;
        margin-bottom: 0.25rem;
    }
    .metric-container .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-container-neutral {
        background-color: rgba(96, 165, 250, 0.8); /* blue-400 */
    }
    .metric-container-cost {
        background-color: rgba(248, 113, 113, 0.8); /* red-400 */
    }
    .metric-container-accuracy-high {
        background-color: rgba(74, 222, 128, 0.8); /* green-400 */
    }
    .metric-container-accuracy-medium {
        background-color: rgba(251, 191, 36, 0.8); /* amber-400 */
    }
    .metric-container-accuracy-low {
        background-color: rgba(248, 113, 113, 0.8); /* red-400 */
    }

    /* General style for sharp corners on Streamlit widgets */
    .stButton>button, 
    .stDownloadButton>button, 
    .stTextInput>div>div>input, 
    .stSelectbox>div>div, 
    .stMultiSelect>div>div, 
    .stDateInput>div>div>input,
    div[data-testid="stExpander"] {
        border-radius: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction with Cognizant branding
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    # Logo and text side by side
    try:
        # Create sub-columns for horizontal alignment with tighter spacing
        logo_subcol, text_subcol = st.columns([1, 2], gap="small")
        
        with logo_subcol:
            st.image("assets/logo.png", width=50)
        
        with text_subcol:
            st.markdown("""
            <div style="padding-top: 8px; margin-left: -20px;">
                <div style="color: #1f77b4; font-weight: bold; font-size: 20px; margin: 0; line-height: 1.2;"></div>
            </div>
            """, unsafe_allow_html=True)
    except:
        # Clean fallback branding - logo only
        st.markdown("""
        <div style="text-align: center; padding: 10px;">
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.title("Prometheus Insights")

with col3:
    # Empty space - removed "Powered by" text
    st.empty()

# Helper functions for custom metric cards
def get_accuracy_color_class(accuracy_value):
    if accuracy_value >= 0.9:
        return "metric-container-accuracy-high"
    elif accuracy_value >= 0.75:
        return "metric-container-accuracy-medium"
    else:
        return "metric-container-accuracy-low"

def create_metric_card(title, value, help_text, color_class):
    st.markdown(f"""
    <div class="metric-container {color_class}" title="{help_text}">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# Sidebar configuration
# st.sidebar.header("Settings")

# Data file paths
DATA_DIR = "data"
MODEL_DATA_FILE = os.path.join(DATA_DIR, "model_data.parquet")
INFRA_DATA_FILE = os.path.join(DATA_DIR, "infra_data.parquet")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def save_data_to_files(model_df, infra_df):
    """Save dataframes to Parquet files for faster loading and smaller file size"""
    try:
        model_df.to_parquet(MODEL_DATA_FILE, index=False)
        infra_df.to_parquet(INFRA_DATA_FILE, index=False)
        return True
    except ImportError:
        # Fallback to CSV if pyarrow is not available
        try:
            csv_model_file = MODEL_DATA_FILE.replace('.parquet', '.csv')
            csv_infra_file = INFRA_DATA_FILE.replace('.parquet', '.csv')
            model_df.to_csv(csv_model_file, index=False)
            infra_df.to_csv(csv_infra_file, index=False)
            st.sidebar.warning("⚠️ Using CSV format (install pyarrow for better performance)")
            return True
        except Exception as e:
            st.error(f"Error saving data files: {e}")
            return False
    except Exception as e:
        st.error(f"Error saving data files: {e}")
        return False

def load_data_from_files():
    """Load dataframes from Parquet files if they exist, fallback to CSV"""
    try:
        # Try Parquet first
        if os.path.exists(MODEL_DATA_FILE) and os.path.exists(INFRA_DATA_FILE):
            model_df = pd.read_parquet(MODEL_DATA_FILE)
            infra_df = pd.read_parquet(INFRA_DATA_FILE)
            
            # Ensure date column is datetime (parquet preserves types better than CSV)
            model_df['date'] = pd.to_datetime(model_df['date'])
            infra_df['date'] = pd.to_datetime(infra_df['date'])
            
            return model_df, infra_df
        
        # Fallback to CSV
        csv_model_file = MODEL_DATA_FILE.replace('.parquet', '.csv')
        csv_infra_file = INFRA_DATA_FILE.replace('.parquet', '.csv')
        
        if os.path.exists(csv_model_file) and os.path.exists(csv_infra_file):
            model_df = pd.read_csv(csv_model_file)
            infra_df = pd.read_csv(csv_infra_file)
            
            # Convert date column back to datetime
            model_df['date'] = pd.to_datetime(model_df['date'])
            infra_df['date'] = pd.to_datetime(infra_df['date'])
            
            return model_df, infra_df
        
        return None, None
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        return None, None

def data_files_exist():
    """Check if data files exist and are not empty"""
    # Check for Parquet files first
    parquet_exist = (os.path.exists(MODEL_DATA_FILE) and os.path.getsize(MODEL_DATA_FILE) > 0 and
                     os.path.exists(INFRA_DATA_FILE) and os.path.getsize(INFRA_DATA_FILE) > 0)
    
    if parquet_exist:
        return True
    
    # Check for CSV fallback files
    csv_model_file = MODEL_DATA_FILE.replace('.parquet', '.csv')
    csv_infra_file = INFRA_DATA_FILE.replace('.parquet', '.csv')
    
    csv_exist = (os.path.exists(csv_model_file) and os.path.getsize(csv_model_file) > 0 and
                 os.path.exists(csv_infra_file) and os.path.getsize(csv_infra_file) > 0)
    
    return csv_exist

# Show modal window
@st.dialog("Remediate")
def show_modal():
  with st.spinner("Connecting to API..."):
      time.sleep(1)
  with st.spinner("Authenticating..."):
      time.sleep(1)
  with st.spinner("Updating..."):
      time.sleep(1)
  with st.spinner("Testing..."):
      time.sleep(1)
  st.success("Done!")
  time.sleep(1)

# Sample data generation function
def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end= pd.Timestamp.today().strftime('%Y-%m-%d'), freq='D')
    
    # Define enterprise-level hierarchy
    cloud_platforms = ['AWS', 'GCP', 'Azure', 'Snowflake', 'Databricks']
    departments = ['Engineering', 'Data Science', 'Marketing', 'Finance', 'Operations']
    environments = ['dev', 'staging', 'prod']
    release_versions = ['v1.0', 'v1.1', 'v1.2', 'v2.0', 'v2.1']
    
    projects_per_cloud = {
        'AWS': ['GenAI-Chat', 'ML-Pipeline', 'Data-Lake'],
        'GCP': ['Analytics-Engine', 'Vision-API', 'Speech-Processing'],
        'Azure': ['Cognitive-Services', 'Bot-Framework', 'Document-AI'],
        'Snowflake': ['Data-Warehouse', 'BI-Analytics', 'ML-Features'],
        'Databricks': ['MLOps-Platform', 'Real-time-Analytics', 'Feature-Store']
    }
    
    avg_tokens_per_project = {
        'GenAI-Chat': 10000, 'ML-Pipeline': 20000, 'Data-Lake': 5000, 'Analytics-Engine': 15000, 'Vision-API': 8000,
        'Speech-Processing': 12000, 'Cognitive-Services': 25000, 'Bot-Framework': 18000, 'Document-AI': 22000,
        'Data-Warehouse': 13000, 'Bot-Framework': 45000,  'BI-Analytics': 60000, 'ML-Features': 65000,
        'MLOps-Platform': 70000, 'Real-time-Analytics': 75000, 'Feature-Store': 80000
    }
    
    token_limits_per_department = {
        'Engineering': 120000, 'Data Science': 100000, 'Marketing': 120000, 'Finance': 130000, 'Operations': 150000 
    }


    # Create project-to-department mapping (each project belongs to only one department)
    project_department_mapping = {}
    dept_index = 0
    for cloud, projects in projects_per_cloud.items():
        for project in projects:
            project_department_mapping[project] = departments[dept_index % len(departments)]
            dept_index += 1
    
    # Model performance data
    models = [ 'GPT-4', 'Claude 3', 'Llama 3', 'Mixtral', 'PaLM 2']

    
    token_limits_per_model_perdepartment = {
        'Engineering': {
            'GPT-4': 1000000,
            'Claude 3': 800000,
            'Llama 3': 900000,
            'Mixtral': 3000000,
            'PaLM 2': 50000000
        },
        'Data Science': {
            'GPT-4': 1000000,
            'Claude 3': 900000,
            'Llama 3': 950000,
            'Mixtral': 3000000,
            'PaLM 2': 50000000
        },
        'Marketing': {
            'GPT-4': 5000000,
            'Claude 3': 1200000, 
            'Llama 3': 2500000,
            'Mixtral': 3000000,
            'PaLM 2': 50000000
        },
        'Finance': {
            'GPT-4': 1500000,
            'Claude 3': 1400000,
            'Llama 3': 2200000,
            'Mixtral': 3000000,
            'PaLM 2': 50000000
        },
        'Operations': {
            'GPT-4': 5000000,
            'Claude 3': 1800000,
            'Llama 3': 1500000,
            'Mixtral': 3000000,
            'PaLM 2': 50000000
        }
    }


    model_data = []
    
    for model in models:
        base_latency = np.random.uniform(100, 500)
        base_throughput = np.random.uniform(100, 5000)
        base_accuracy = np.random.uniform(0.7, 0.95)
        base_cost = np.random.uniform(0.01, 0.1)
        
        for date in dates:
            # Generate combinations for each model
            for cloud in cloud_platforms:
                for project in projects_per_cloud[cloud]:
                    # Get the department for this project (one-to-one mapping)
                    dept = project_department_mapping[project]
                    for env in environments:
                        # Skip some combinations to make data more realistic
                        if np.random.random() > 0.3:  # 70% of combinations exist
                            continue
                            
                        release_version = np.random.choice(release_versions)
                        
                        # Add some trend and random noise
                        trend_factor = 1 - (0.0001 * (date - dates[0]).days)  # Gradual improvement over time
                        random_factor = np.random.normal(1, 0.05)  # Daily variation
                        
                        # Cloud-specific cost factors
                        cloud_cost_factor = {'AWS': 1.0, 'GCP': 0.9, 'Azure': 1.1, 'Snowflake': 1.3, 'Databricks': 1.2}[cloud]
                        # Environment-specific factors
                        env_factor = {'dev': 0.3, 'staging': 0.6, 'prod': 1.0}[env]
                        
                        latency = base_latency * trend_factor * random_factor * env_factor
                        throughput = base_throughput / trend_factor * random_factor * env_factor
                        accuracy = min(0.99, base_accuracy / trend_factor * random_factor)
                        cost = base_cost * trend_factor * random_factor * cloud_cost_factor * env_factor
                        
                        # Token usage simulation
                        avg_tokens_per_minute = (throughput / (60 * 24)) * avg_tokens_per_project[project] * np.random.normal(1, 0.15)
                        max_tokens_per_minute = avg_tokens_per_minute * np.random.uniform(1.5, 7.5)
                        token_limit = token_limits_per_model_perdepartment[dept][model]
                        exceptions = np.sum(max_tokens_per_minute > token_limit)
            
                        model_data.append({
                            'date': date,
                            'model': model,
                            'cloud_platform': cloud,
                            'project': project,
                            'department': dept,
                            'environment': env,
                            'release_version': release_version,
                            'latency_ms': latency,
                            'throughput_qps': throughput,
                            'accuracy': accuracy,
                            'cost_per_1k_tokens': cost,
                            'memory_usage_gb': np.random.uniform(4, 16) * env_factor,
                            'gpu_utilization': np.random.uniform(0.4, 0.95) * env_factor,
                            'avg_tokens_per_minute': avg_tokens_per_minute,
                            'max_tokens_per_minute': max_tokens_per_minute,
                            'token_limit_exception_count': exceptions,
                            'token_limit': token_limit
                        })
    
    # Infrastructure data
    infra_data = []
    components = ['API Gateway', 'Load Balancer', 'Model Server', 'Cache', 'Database']
    
    for component in components:
        base_cpu = np.random.uniform(20, 60)
        base_memory = np.random.uniform(30, 70)
        base_requests = np.random.uniform(100, 1000)
        
        for date in dates:
            for cloud in cloud_platforms:
                for project in projects_per_cloud[cloud]:
                    # Get the department for this project (one-to-one mapping)
                    dept = project_department_mapping[project]
                    for env in environments:
                        # Skip some combinations to make data more realistic
                        if np.random.random() > 0.4:  # 60% of combinations exist
                            continue
                            
                        release_version = np.random.choice(release_versions)
                        
                        # Weekly pattern + trend
                        day_of_week = date.dayofweek
                        weekly_factor = 1 + 0.2 * (day_of_week < 5)  # Higher on weekdays
                        trend_factor = 1 + (0.0005 * (date - dates[0]).days)  # Gradual increase in load
                        
                        # Environment-specific factors
                        env_factor = {'dev': 0.3, 'staging': 0.6, 'prod': 1.0}[env]
                        
                        cpu_usage = base_cpu * weekly_factor * trend_factor * np.random.normal(1, 0.1) * env_factor
                        memory_usage = base_memory * weekly_factor * trend_factor * np.random.normal(1, 0.05) * env_factor
                        requests = base_requests * weekly_factor * trend_factor * np.random.normal(1, 0.2) * env_factor
                        
                        infra_data.append({
                            'date': date,
                            'component': component,
                            'cloud_platform': cloud,
                            'project': project,
                            'department': dept,
                            'environment': env,
                            'release_version': release_version,
                            'cpu_usage_percent': min(100, cpu_usage),
                            'memory_usage_percent': min(100, memory_usage),
                            'requests_per_minute': requests,
                            'errors_per_minute': requests * np.random.uniform(0.001, 0.05),
                            'avg_response_time_ms': np.random.uniform(50, 500)
                        })
    
    # Create DataFrames
    model_df = pd.DataFrame(model_data)
    infra_df = pd.DataFrame(infra_data)
    
    return model_df, infra_df

# Load or generate data with persistent storage
def load_or_generate_data():
    """Load data from files or generate new data if files don't exist"""
    
    # Add a button to regenerate data (useful for development/testing)
    if st.sidebar.button("Regenerate Data", help="Generate new sample data and overwrite existing files"):
        with st.spinner('Generating new sample data for all services...'):
            # Regenerate model and infra data
            model_df, infra_df = generate_sample_data()
            save_data_to_files(model_df, infra_df)
            
            # Regenerate billing data
            mock_billing_service.load_or_generate_billing_data(force_regenerate=True)
            
            st.sidebar.success("All data regenerated!")
            return model_df, infra_df
    
    # Try to load existing data first
    if data_files_exist():
        with st.spinner('Loading data from files...'):
            model_df, infra_df = load_data_from_files()
            if model_df is not None and infra_df is not None:
                return model_df, infra_df
    
    # Generate new data if files don't exist or loading failed
    with st.spinner('Generating sample data (first time setup)...'):
        model_df, infra_df = generate_sample_data()
        
        # Save the generated data for future use
        if save_data_to_files(model_df, infra_df):
            st.sidebar.success("Data generated and saved for faster future loading!")
        else:
            st.sidebar.warning("Data generated but couldn't save to files")
        
        return model_df, infra_df

# Load data using the new persistent approach
model_df, infra_df = load_or_generate_data()

# Calculate error rate on the main dataframe to ensure it's always available
infra_df['error_rate'] = infra_df['errors_per_minute'] / infra_df['requests_per_minute']

# Sidebar: Data Management Section
st.sidebar.markdown("---")
st.sidebar.markdown("### Data Management")

# Show data freshness info
if data_files_exist():
    model_mtime = os.path.getmtime(MODEL_DATA_FILE)
    infra_mtime = os.path.getmtime(INFRA_DATA_FILE)
    last_updated = datetime.fromtimestamp(max(model_mtime, infra_mtime))
    st.sidebar.text(f"Last updated: {last_updated.strftime('%Y-%m-%d %H:%M')}")
    
    # Option to clear data files
    if st.sidebar.button("Clear Data Files", help="Delete saved data files (will regenerate on next load)"):
        try:
            files_deleted = 0
            # Remove Parquet files
            if os.path.exists(MODEL_DATA_FILE):
                os.remove(MODEL_DATA_FILE)
                files_deleted += 1
            if os.path.exists(INFRA_DATA_FILE):
                os.remove(INFRA_DATA_FILE)
                files_deleted += 1
            
            # Remove CSV fallback files
            csv_model_file = MODEL_DATA_FILE.replace('.parquet', '.csv')
            csv_infra_file = INFRA_DATA_FILE.replace('.parquet', '.csv')
            if os.path.exists(csv_model_file):
                os.remove(csv_model_file)
                files_deleted += 1
            if os.path.exists(csv_infra_file):
                os.remove(csv_infra_file)
                files_deleted += 1
                
            st.sidebar.success(f"{files_deleted} data files cleared!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error clearing files: {e}")

st.sidebar.markdown("---")

# Add sidebar drill-down filters

# Initialize session state for filter persistence and reset functionality
if 'filter_reset' not in st.session_state:
    st.session_state.filter_reset = False

# Handle filter reset
if st.session_state.filter_reset:
    # Clear all filter-related session state
    filter_keys = ['cloud_filter', 'dept_filter', 'project_filter', 'env_filter', 'release_filter', 'model_filter', 'component_filter', 'metrics_filter']
    for key in filter_keys:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.filter_reset = False
    st.rerun()

# Date filter for analysis (needs to come first to filter other options)
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(model_df['date'].min().date(), model_df['date'].max().date()),
    min_value=model_df['date'].min().date(),
    max_value=model_df['date'].max().date(),
    key="date_filter"
)

# Apply date filter first for filtering options
date_filtered_model_df = model_df.copy()
date_filtered_infra_df = infra_df.copy()
if len(date_range) == 2:
    start_date, end_date = date_range
    date_filtered_model_df = date_filtered_model_df[(date_filtered_model_df['date'].dt.date >= start_date) & (date_filtered_model_df['date'].dt.date <= end_date)]
    date_filtered_infra_df = date_filtered_infra_df[(date_filtered_infra_df['date'].dt.date >= start_date) & (date_filtered_infra_df['date'].dt.date <= end_date)]

# Get current selections from session state (if they exist)
current_clouds = st.session_state.get('cloud_filter', [])
current_departments = st.session_state.get('dept_filter', [])
current_projects = st.session_state.get('project_filter', [])
current_environments = st.session_state.get('env_filter', [])
current_releases = st.session_state.get('release_filter', [])

# Create a combined filter to determine what options should be available for each filter
# Start with date-filtered data
working_model_df = date_filtered_model_df.copy()
working_infra_df = date_filtered_infra_df.copy()

# Apply all existing filters except the one we're currently determining options for
def get_intersected_data(exclude_filter=None):
    temp_model_df = date_filtered_model_df.copy()
    temp_infra_df = date_filtered_infra_df.copy()
    
    if exclude_filter != 'cloud' and current_clouds:
        temp_model_df = temp_model_df[temp_model_df['cloud_platform'].isin(current_clouds)]
        temp_infra_df = temp_infra_df[temp_infra_df['cloud_platform'].isin(current_clouds)]
    
    if exclude_filter != 'department' and current_departments:
        temp_model_df = temp_model_df[temp_model_df['department'].isin(current_departments)]
        temp_infra_df = temp_infra_df[temp_infra_df['department'].isin(current_departments)]
    
    if exclude_filter != 'project' and current_projects:
        temp_model_df = temp_model_df[temp_model_df['project'].isin(current_projects)]
        temp_infra_df = temp_infra_df[temp_infra_df['project'].isin(current_projects)]
    
    if exclude_filter != 'environment' and current_environments:
        temp_model_df = temp_model_df[temp_model_df['environment'].isin(current_environments)]
        temp_infra_df = temp_infra_df[temp_infra_df['environment'].isin(current_environments)]
    
    if exclude_filter != 'release' and current_releases:
        temp_model_df = temp_model_df[temp_model_df['release_version'].isin(current_releases)]
        temp_infra_df = temp_infra_df[temp_infra_df['release_version'].isin(current_releases)]
    
    return temp_model_df, temp_infra_df

# Cloud Platform filter - shows clouds that have data for other selected filters
cloud_data, _ = get_intersected_data('cloud')
available_clouds = sorted(cloud_data['cloud_platform'].unique()) if len(cloud_data) > 0 else []
selected_clouds = st.sidebar.multiselect(
    "Cloud Platforms",
    options=available_clouds,
    default=[c for c in current_clouds if c in available_clouds] if current_clouds else available_clouds,
    help=f"Cloud platforms with data in current selection scope ({len(available_clouds)} available)",
    key="cloud_filter"
)

# Department filter - shows departments that exist with current cloud/project/env/release selections
dept_data, _ = get_intersected_data('department')
available_departments = sorted(dept_data['department'].unique()) if len(dept_data) > 0 else []
selected_departments = st.sidebar.multiselect(
    "Departments",
    options=available_departments,
    default=[d for d in current_departments if d in available_departments] if current_departments else available_departments,
    help=f"Departments with data in current selection scope ({len(available_departments)} available)",
    key="dept_filter"
)

# Project filter - shows projects that exist with current cloud/dept/env/release selections
project_data, _ = get_intersected_data('project')
available_projects = sorted(project_data['project'].unique()) if len(project_data) > 0 else []
selected_projects = st.sidebar.multiselect(
    "Projects",
    options=available_projects,
    default=[p for p in current_projects if p in available_projects] if current_projects else available_projects,
    help=f"Projects with data in current selection scope ({len(available_projects)} available)",
    key="project_filter"
)

# Environment filter - shows environments that exist with current cloud/dept/project/release selections
env_data, _ = get_intersected_data('environment')
available_environments = sorted(env_data['environment'].unique()) if len(env_data) > 0 else []
selected_environments = st.sidebar.multiselect(
    "Environments",
    options=available_environments,
    default=[e for e in current_environments if e in available_environments] if current_environments else available_environments,
    help=f"Environments with data in current selection scope ({len(available_environments)} available)",
    key="env_filter"
)

# Release Version filter - shows releases that exist with current cloud/dept/project/env selections
release_data, _ = get_intersected_data('release')
available_releases = sorted(release_data['release_version'].unique()) if len(release_data) > 0 else []
selected_releases = st.sidebar.multiselect(
    "Release Versions",
    options=available_releases,
    default=[r for r in current_releases if r in available_releases] if current_releases else available_releases,
    help=f"Release versions with data in current selection scope ({len(available_releases)} available)",
    key="release_filter"
)

# Apply all selected filters to create final filtered datasets
final_model_df = date_filtered_model_df.copy()
final_infra_df = date_filtered_infra_df.copy()

if selected_clouds:
    final_model_df = final_model_df[final_model_df['cloud_platform'].isin(selected_clouds)]
    final_infra_df = final_infra_df[final_infra_df['cloud_platform'].isin(selected_clouds)]

if selected_departments:
    final_model_df = final_model_df[final_model_df['department'].isin(selected_departments)]
    final_infra_df = final_infra_df[final_infra_df['department'].isin(selected_departments)]

if selected_projects:
    final_model_df = final_model_df[final_model_df['project'].isin(selected_projects)]
    final_infra_df = final_infra_df[final_infra_df['project'].isin(selected_projects)]

if selected_environments:
    final_model_df = final_model_df[final_model_df['environment'].isin(selected_environments)]
    final_infra_df = final_infra_df[final_infra_df['environment'].isin(selected_environments)]

if selected_releases:
    final_model_df = final_model_df[final_model_df['release_version'].isin(selected_releases)]
    final_infra_df = final_infra_df[final_infra_df['release_version'].isin(selected_releases)]

# Use the final filtered data as our base filtered datasets
model_df_filtered = final_model_df.copy()
infra_df_filtered = final_infra_df.copy()

# Display current filter summary with counts and bidirectional filter impact
st.sidebar.markdown("### 📊 Current Selection")

# Calculate available options at each level given current selection
cloud_options_data, _ = get_intersected_data('cloud')
dept_options_data, _ = get_intersected_data('department')
project_options_data, _ = get_intersected_data('project')
env_options_data, _ = get_intersected_data('environment')
release_options_data, _ = get_intersected_data('release')

# Show available vs selected for each filter level
total_clouds = len(sorted(cloud_options_data['cloud_platform'].unique())) if len(cloud_options_data) > 0 else 0
total_depts = len(sorted(dept_options_data['department'].unique())) if len(dept_options_data) > 0 else 0
total_projects = len(sorted(project_options_data['project'].unique())) if len(project_options_data) > 0 else 0
total_envs = len(sorted(env_options_data['environment'].unique())) if len(env_options_data) > 0 else 0
total_releases = len(sorted(release_options_data['release_version'].unique())) if len(release_options_data) > 0 else 0

# Model selection (based on all filtered data from enterprise filters)
available_models = sorted(model_df_filtered['model'].unique()) if len(model_df_filtered) > 0 else []
selected_models = st.sidebar.multiselect(
    f"Select Models ({len(available_models)} available)",
    options=available_models,
    default=available_models[:3] if len(available_models) > 0 else [],  # Default select first 3 models
    help=f"Models available in current enterprise scope. Selecting specific models may affect infrastructure component availability.",
    key="model_filter"
)

# Apply model filter and show impact
if selected_models:
    model_filtered_df = model_df_filtered[model_df_filtered['model'].isin(selected_models)]
else:
    model_filtered_df = model_df_filtered
    if len(available_models) > 0:
        st.sidebar.warning("⚠️ No models selected - showing empty dataset")

# Infrastructure component selection (based on filtered data)
available_components = sorted(infra_df_filtered['component'].unique()) if len(infra_df_filtered) > 0 else []
selected_components = st.sidebar.multiselect(
    f"Infrastructure Components ({len(available_components)} available)",
    options=available_components,
    default=available_components,
    help=f"Infrastructure components in current enterprise scope.",
    key="component_filter"
)

# Apply component filter and show impact
if selected_components:
    infra_filtered_df = infra_df_filtered[infra_df_filtered['component'].isin(selected_components)]
else:
    infra_filtered_df = infra_df_filtered
    if len(available_components) > 0:
        st.sidebar.warning("⚠️ No components selected - showing empty dataset")

# Metric selection for models
model_metrics = st.sidebar.multiselect(
    "Select Model Metrics to Analyze",
    options=['latency_ms', 'throughput_qps', 'accuracy', 'cost_per_1k_tokens', 'memory_usage_gb', 'gpu_utilization'],
    default=['latency_ms', 'accuracy', 'cost_per_1k_tokens'],
    help="Choose which model performance metrics to display",
    key="metrics_filter"
)

# Final filter application - use the progressively filtered datasets
model_df_filtered = model_filtered_df if selected_models else model_df_filtered.iloc[0:0]  # Empty if no models selected
infra_df_filtered = infra_filtered_df if selected_components else infra_df_filtered.iloc[0:0]  # Empty if no components selected

# Add a reset filters button
if st.sidebar.button("Reset All Filters", help="Reset all filters to show all available data"):
    st.session_state.filter_reset = True
    st.rerun()

st.markdown("---")

# Main content with Summary tab as first tab
tab_summary, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Summary", "Model Performance", "Infrastructure Metrics", "Architecture Overview", "Optimization Recommendations", "Token Usage Analysis", "Alerts", "Billing Details"])

with tab_summary:
    st.header("Analysis Summary & Context")
    
    # Show dynamic filter context in main area
    st.markdown("### Current Analysis Context")

    if len(model_df_filtered) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card("Departments", len(selected_departments), "Number of departments in current selection", "metric-container-neutral")
        with col2:
            create_metric_card("Cloud Platforms", len(selected_clouds), "Number of cloud platforms selected", "metric-container-neutral")
        with col3:
            create_metric_card("Projects", len(selected_projects), "Number of projects in current scope", "metric-container-neutral")
        with col4:
            create_metric_card("Environments", len(selected_environments), "Number of environments (dev/staging/prod)", "metric-container-neutral")
        
        # Show data volume metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            create_metric_card("Model Records", f"{len(model_df_filtered):,}", "Number of model performance records in current selection", "metric-container-neutral")
        with col2:
            create_metric_card("Infra Records", f"{len(infra_df_filtered):,}", "Number of infrastructure records in current selection", "metric-container-neutral")
        with col3:
            total_cost = model_df_filtered['cost_per_1k_tokens'].sum() if len(model_df_filtered) > 0 else 0
            create_metric_card("Total Cost", f"${total_cost:,.2f}", "Sum of all costs in current selection", "metric-container-cost")
        with col4:
            avg_accuracy = model_df_filtered['accuracy'].mean() if len(model_df_filtered) > 0 else 0
            accuracy_color_class = get_accuracy_color_class(avg_accuracy)
            create_metric_card("Avg Accuracy", f"{avg_accuracy:.1%}", "Average model accuracy in current selection", accuracy_color_class)
    else:
        st.warning("No data matches your current filter selection. Try adjusting your filters.")
        st.info("**Tip**: Use the sidebar filters to drill down into specific clouds, departments, projects, environments, or models.")

    # Show breakdown by cloud and environment (if data exists)
    if len(model_df_filtered) > 0:
        st.markdown("### Current Selection Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost by cloud platform
            cost_by_cloud = model_df_filtered.groupby('cloud_platform')['cost_per_1k_tokens'].mean().sort_values(ascending=False)
            fig_cloud_cost = px.bar(
                x=cost_by_cloud.index, 
                y=cost_by_cloud.values,
                title="Average Cost per 1K Tokens by Cloud Platform",
                labels={'x': 'Cloud Platform', 'y': 'Avg Cost ($)'}
            )
            st.plotly_chart(fig_cloud_cost, use_container_width=True)
        
        with col2:
            # Projects by environment
            env_dist = model_df_filtered.groupby(['environment', 'cloud_platform']).size().reset_index(name='count')
            fig_env_dist = px.bar(
                env_dist, 
                x='environment', 
                y='count', 
                color='cloud_platform',
                title="Data Points by Environment & Cloud",
                labels={'count': 'Number of Records', 'environment': 'Environment'}
            )
            st.plotly_chart(fig_env_dist, use_container_width=True)
        
        # Additional summary chart
        st.markdown("### Key Performance Insights")
        
        # Average cost by release version
        release_cost = model_df_filtered.groupby('release_version')['cost_per_1k_tokens'].mean().sort_values(ascending=True)
        fig_release_cost = px.bar(
            x=release_cost.index,
            y=release_cost.values,
            title="Average Cost per 1K Tokens by Release Version",
            labels={'x': 'Release Version', 'y': 'Cost per 1K Tokens ($)'}
        )
        st.plotly_chart(fig_release_cost, use_container_width=True)

with tab1:
    st.header("Model Performance Analysis")
    
    # Model performance comparison - latest metrics
    st.subheader("Model Comparison - Current Performance")
    
    latest_date = model_df_filtered['date'].max()
    latest_metrics = model_df_filtered[model_df_filtered['date'] == latest_date]
    
    # Create a radar chart for model comparison
    if not latest_metrics.empty and len(selected_models) > 0:
        fig = go.Figure()
        
        for model in selected_models:
            model_data = latest_metrics[latest_metrics['model'] == model]
            if not model_data.empty:
                # Normalize metrics for radar chart
                metrics_to_plot = {
                    'Latency (lower is better)': 1 - (model_data['latency_ms'].values[0] / latest_metrics['latency_ms'].max()),
                    'Throughput': model_data['throughput_qps'].values[0] / latest_metrics['throughput_qps'].max(),
                    'Accuracy': model_data['accuracy'].values[0],
                    'Cost Efficiency (lower is better)': 1 - (model_data['cost_per_1k_tokens'].values[0] / latest_metrics['cost_per_1k_tokens'].max()),
                    'Memory Efficiency (lower is better)': 1 - (model_data['memory_usage_gb'].values[0] / latest_metrics['memory_usage_gb'].max()),
                    'GPU Utilization': model_data['gpu_utilization'].values[0]
                }
                
                fig.add_trace(go.Scatterpolar(
                    r=list(metrics_to_plot.values()),
                    theta=list(metrics_to_plot.keys()),
                    fill='toself',
                    name=model
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            height=500,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Metric trends over time
    st.subheader("Model Metric Trends Over Time")

    # Graph type selector for model metric trends
    model_graph_type = st.radio(
        "Select Graph Type for Model Metrics",
        options=["Line", "Bar", "Area"],
        horizontal=True,
        key="model_metric_graph_type"
    )

    cols = st.columns(len(model_metrics) if len(model_metrics) > 0 else 1)

    for i, metric in enumerate(model_metrics):
        with cols[i % len(cols)]:
            pretty_metric = metric.replace('_', ' ').title()
            st.write(f"**{pretty_metric}**")

            if model_graph_type == "Line":
                fig = px.line(
                    model_df_filtered, 
                    x='date', 
                    y=metric, 
                    color='model',
                    title=f"{pretty_metric} Over Time"
                )
            elif model_graph_type == "Bar":
                fig = px.bar(
                    model_df_filtered, 
                    x='date', 
                    y=metric, 
                    color='model',
                    barmode='group',
                    title=f"{pretty_metric} Over Time"
                )
            elif model_graph_type == "Area":
                fig = px.area(
                    model_df_filtered, 
                    x='date', 
                    y=metric, 
                    color='model',
                    title=f"{pretty_metric} Over Time"
                )

            # Add a trend line for each model (only for line chart)
            if model_graph_type == "Line":
                for model in selected_models:
                    model_data = model_df_filtered[model_df_filtered['model'] == model]
                    if len(model_data) > 1:  # Need at least 2 points for a trendline
                        fig.add_traces(
                            px.scatter(model_data, x='date', y=metric, trendline='lowess').data[1]
                        )

            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Infrastructure Metrics")
    
    # Component load over time
    st.subheader("Component Load Over Time")
    
    metric_options = ['cpu_usage_percent', 'memory_usage_percent', 'requests_per_minute', 'errors_per_minute', 'avg_response_time_ms']
    selected_infra_metric = st.selectbox("Select Metric", metric_options, index=0)
    
    # Graph type selector for component load
    graph_type = st.radio(
        "Select Graph Type for Component Load",
        options=["Line", "Bar"],
        horizontal=True,
        key="component_load_graph_type"
    )
    
    if graph_type == "Line":
        fig = px.line(
            infra_df_filtered, 
            x='date', 
            y=selected_infra_metric, 
            color='component',
            title=f"{selected_infra_metric.replace('_', ' ').title()} Over Time"
        )
    else:
        fig = px.bar(
            infra_df_filtered, 
            x='date', 
            y=selected_infra_metric, 
            color='component',
            barmode='group',
            title=f"{selected_infra_metric.replace('_', ' ').title()} Over Time"
        )
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap between metrics
    st.subheader("Correlation Between Infrastructure Metrics")
    
    # Graph type selector for correlation
    corr_graph_type = st.radio(
        "Select Graph Type for Correlation",
        options=["Heatmap", "Bar", "Line"],
        horizontal=True,
        key="correlation_graph_type"
    )
    
    # Calculate average metrics by component
    avg_metrics_by_component = infra_df_filtered.groupby('component')[metric_options].mean().reset_index()
    correlation_matrix = avg_metrics_by_component[metric_options].corr()
    
    if corr_graph_type == "Heatmap":
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        plt.title('Correlation Between Infrastructure Metrics')
        st.pyplot(fig)
    elif corr_graph_type == "Bar":
        # Show correlation of each metric with the first metric as a bar chart
        first_metric = metric_options[0]
        corr_with_first = correlation_matrix[first_metric].drop(first_metric)
        fig = px.bar(
            x=corr_with_first.index,
            y=corr_with_first.values,
            labels={'x': 'Metric', 'y': f'Correlation with {first_metric}'},
            title=f'Correlation of Metrics with {first_metric}'
        )
        st.plotly_chart(fig, use_container_width=True)
    elif corr_graph_type == "Line":
        # Show all correlations as lines
        fig = go.Figure()
        for metric in metric_options:
            fig.add_trace(go.Scatter(
                x=correlation_matrix.columns,
                y=correlation_matrix[metric],
                mode='lines+markers',
                name=metric
            ))
        fig.update_layout(
            title='Correlation Lines Between Infrastructure Metrics',
            xaxis_title='Metric',
            yaxis_title='Correlation',
            legend_title='Metric'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Error rate analysis
    st.subheader("Error Rate Analysis")
    
    fig = px.box(
        infra_df_filtered, 
        x='component', 
        y='error_rate',
        color='component',
        title="Error Rate Distribution by Component"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Architecture Overview")
    
    # Create a simple architecture diagram using Graphviz
    st.subheader("Current Architecture Diagram")
    
    # Generate a simple architecture diagram
    architecture_diagram = """
    digraph G {
        rankdir=LR;
        
        subgraph cluster_user {
            label="User Layer";
            User [shape=circle];
            WebUI [shape=box];
            API [shape=box];
            User -> WebUI;
            WebUI -> API;
        }
        
        subgraph cluster_gateway {
            label="API Gateway";
            Gateway [shape=box];
            LoadBalancer [shape=box];
            API -> Gateway;
            Gateway -> LoadBalancer;
        }
        
        subgraph cluster_serving {
            label="Serving Layer";
            ModelServer [shape=box];
            Cache [shape=cylinder];
            LoadBalancer -> ModelServer;
            ModelServer -> Cache [dir=both];
        }
        
        subgraph cluster_models {
            label="Model Layer";
            GPT4 [shape=box];
            Claude [shape=box];
            Llama [shape=box];
            Mixtral [shape=box];
            PaLM [shape=box];
            ModelServer -> GPT4;
            ModelServer -> Claude;
            ModelServer -> Llama;
            ModelServer -> Mixtral;
            ModelServer -> PaLM;
        }
        
        subgraph cluster_storage {
            label="Storage Layer";
            Database [shape=cylinder];
            ModelServer -> Database [dir=both];
        }
        
        subgraph cluster_monitoring {
            label="Monitoring Layer";
            Metrics [shape=box];
            Logging [shape=box];
            ModelServer -> Metrics [style=dashed];
            Gateway -> Metrics [style=dashed];
            Database -> Logging [style=dashed];
            ModelServer -> Logging [style=dashed];
        }
    }
    """
    
    st.graphviz_chart(architecture_diagram)
    
    # Resource allocation analysis
    st.subheader("Resource Allocation Analysis")

    # Graph type selector for resource allocation
    resource_graph_type = st.radio(
        "Select Graph Type for Resource Allocation",
        options=["Horizontal Bar", "Pie"],
        horizontal=True,
        key="resource_allocation_graph_type"
    )

    # Calculate average resource usage by component
    avg_resources = infra_df_filtered.groupby('component')[['cpu_usage_percent', 'memory_usage_percent']].mean().reset_index()

    if resource_graph_type == "Horizontal Bar":
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=avg_resources['component'],
            x=avg_resources['cpu_usage_percent'],
            name='CPU Usage (%)',
            orientation='h',
            marker=dict(color='rgba(58, 71, 80, 0.6)')
        ))
        fig.add_trace(go.Bar(
            y=avg_resources['component'],
            x=avg_resources['memory_usage_percent'],
            name='Memory Usage (%)',
            orientation='h',
            marker=dict(color='rgba(246, 78, 139, 0.6)')
        ))
        fig.update_layout(
            barmode='group',
            title='Average Resource Usage by Component',
            xaxis_title='Usage Percentage',
            yaxis_title='Component',
            legend_title='Resource Type'
        )
        st.plotly_chart(fig, use_container_width=True)
    elif resource_graph_type == "Pie":
        # Show CPU and Memory as two pie charts
        pie_cols = st.columns(2)
        with pie_cols[0]:
            fig_cpu = px.pie(avg_resources, values='cpu_usage_percent', names='component', title='CPU Usage Distribution')
            st.plotly_chart(fig_cpu, use_container_width=True)
        with pie_cols[1]:
            fig_mem = px.pie(avg_resources, values='memory_usage_percent', names='component', title='Memory Usage Distribution')
            st.plotly_chart(fig_mem, use_container_width=True)
    
    # Calculate bottlenecks
    st.subheader("Potential Bottlenecks")
    
    # Find components with high resource usage
    high_usage = infra_df_filtered.groupby('component').agg({
        'cpu_usage_percent': ['mean', 'max'],
        'memory_usage_percent': ['mean', 'max'],
        'avg_response_time_ms': ['mean', 'max']
    }).reset_index()
    
    high_usage.columns = ['component', 'cpu_mean', 'cpu_max', 'memory_mean', 'memory_max', 'response_mean', 'response_max']
    
    # Define bottleneck thresholds
    high_usage['cpu_bottleneck'] = high_usage['cpu_max'] > 80
    high_usage['memory_bottleneck'] = high_usage['memory_max'] > 80
    high_usage['response_bottleneck'] = high_usage['response_max'] > 300
    
    # Create bottleneck dataframe
    bottlenecks = high_usage[['component', 'cpu_mean', 'cpu_max', 'memory_mean', 'memory_max', 'response_mean', 'response_max']]
    bottlenecks = bottlenecks.sort_values(by=['cpu_max', 'memory_max', 'response_max'], ascending=False)
    
    st.dataframe(
        bottlenecks,
        column_config={
            "component": "Component",
            "cpu_mean": st.column_config.NumberColumn("Avg CPU %", format="%.1f"),
            "cpu_max": st.column_config.NumberColumn("Max CPU %", format="%.1f"),
            "memory_mean": st.column_config.NumberColumn("Avg Memory %", format="%.1f"),
            "memory_max": st.column_config.NumberColumn("Max Memory %", format="%.1f"),
            "response_mean": st.column_config.NumberColumn("Avg Response (ms)", format="%.1f"),
            "response_max": st.column_config.NumberColumn("Max Response (ms)", format="%.1f"),
        },
        use_container_width=True,
        hide_index=True,
    )

with tab4:
    st.header("Optimization Recommendations")
    
    # --- Data Processing for Recommendations ---
    
    # Calculate model efficiency scores
    latest_metrics_raw = model_df_filtered[model_df_filtered['date'] == model_df_filtered['date'].max()]
    model_recommendations = pd.DataFrame()
    
    if not latest_metrics_raw.empty:
        latest_metrics = latest_metrics_raw.copy()

        def normalize(series, higher_is_better=True):
            min_val, max_val = series.min(), series.max()
            if max_val == min_val:
                return 0.5
            normalized = (series - min_val) / (max_val - min_val)
            return normalized if higher_is_better else 1 - normalized

        # Normalize metrics for scoring
        latest_metrics['latency_norm'] = normalize(latest_metrics['latency_ms'], higher_is_better=False)
        latest_metrics['throughput_norm'] = normalize(latest_metrics['throughput_qps'], higher_is_better=True)
        latest_metrics['accuracy_norm'] = normalize(latest_metrics['accuracy'], higher_is_better=True)
        latest_metrics['cost_norm'] = normalize(latest_metrics['cost_per_1k_tokens'], higher_is_better=False)
        latest_metrics['memory_norm'] = normalize(latest_metrics['memory_usage_gb'], higher_is_better=False)
        
        # Calculate efficiency score (higher is better)
        latest_metrics['efficiency_score'] = (
            latest_metrics['latency_norm'] * 0.25 +
            latest_metrics['throughput_norm'] * 0.2 +
            latest_metrics['accuracy_norm'] * 0.3 +
            latest_metrics['cost_norm'] * 0.15 +
            latest_metrics['memory_norm'] * 0.1
        )
        
        # Sort by efficiency score and include contextual columns
        model_recommendations = latest_metrics[['model', 'cloud_platform', 'department', 'project', 'environment', 'release_version', 'efficiency_score', 'latency_ms', 'throughput_qps', 'accuracy', 'cost_per_1k_tokens']].sort_values('efficiency_score', ascending=False)

    # Generate recommendations based on data analysis
    recommendations = []
    
    # Check for model-specific recommendations
    if not model_recommendations.empty:
        best_model = model_recommendations.iloc[0]['model']
        recommendations.append({'severity': 'low', 'text': f"🔹 Consider using **{best_model}** as your primary model based on overall efficiency score."})
        
        cost_efficient = model_recommendations.sort_values('cost_per_1k_tokens').iloc[0]['model']
        if cost_efficient != best_model:
            recommendations.append({'severity': 'medium', 'text': f"🔹 For cost-sensitive applications, **{cost_efficient}** provides the best value."})
            
        low_latency = model_recommendations.sort_values('latency_ms').iloc[0]['model']
        if low_latency != best_model:
            recommendations.append({'severity': 'medium', 'text': f"🔹 For latency-critical applications, **{low_latency}** provides the fastest response times."})

    # Check for infrastructure recommendations
    if not bottlenecks.empty:
        cpu_bottlenecks = bottlenecks[bottlenecks['cpu_max'] > 80]['component'].tolist()
        if cpu_bottlenecks:
            recommendations.append({'severity': 'high', 'text': f"🔹 Consider scaling up or out the following components with high CPU usage: **{', '.join(cpu_bottlenecks)}**."})
        
        memory_bottlenecks = bottlenecks[bottlenecks['memory_max'] > 80]['component'].tolist()
        if memory_bottlenecks:
            recommendations.append({'severity': 'high', 'text': f"🔹 Increase memory allocation for: **{', '.join(memory_bottlenecks)}**."})
            
        response_bottlenecks = bottlenecks[bottlenecks['response_max'] > 300]['component'].tolist()
        if response_bottlenecks:
            recommendations.append({'severity': 'high', 'text': f"🔹 Optimize or scale the following components to reduce response times: **{', '.join(response_bottlenecks)}**."})

    # Error rate recommendations
    if not infra_df_filtered.empty and 'error_rate' in infra_df_filtered.columns:
        high_error_components = infra_df_filtered.groupby('component')['error_rate'].mean()
        high_error_components = high_error_components[high_error_components > 0.01].index.tolist()
        if high_error_components:
            recommendations.append({'severity': 'medium', 'text': f"🔹 Investigate and reduce error rates in: **{', '.join(high_error_components)}**."})

    # Cache optimization recommendations
    cache_data = infra_df_filtered[infra_df_filtered['component'] == 'Cache']
    if not cache_data.empty and cache_data['cpu_usage_percent'].mean() > 60:
        recommendations.append({'severity': 'medium', 'text': "🔹 Consider implementing a more efficient caching strategy or scaling your cache layer."})

    # Architecture recommendations
    recommendations.append({'severity': 'low', 'text': "🔹 Consider implementing a model router to dynamically select the optimal model based on request characteristics."})
    recommendations.append({'severity': 'low', 'text': "🔹 Add redundancy to critical components to improve system reliability."})

    # --- UI Display ---

    # --- Session state for checks_df ---
    if 'checks_df' not in st.session_state:
        checks_list = [
            "Budget implemented", "Cost alert implemented", "Token alert implemented",
            "Storage alert implemented", "Compute alert implemented", "Labels implemented",
            "Chunk size based on content type check", "Multi agent implemented",
            "Agent payload watch implemented", "Semantic caching implemented",
            "GPTCache implemented", "Langchain caching implemented",
            "Agent feedback implemented", "Model size check", "Model version check",
            "Subscription check", "Anomaly detection check"
        ]
        priorities = np.random.choice(['High', 'Medium', 'Low'], size=len(checks_list))
        action_values = np.random.choice(['Remediate', 'Done', 'N/A'], size=len(checks_list))
        st.session_state.checks_df = pd.DataFrame({
            'Check': checks_list,
            'Priority': priorities,
            'Action': action_values
        })

    checks_df = st.session_state.checks_df

    # Calculate counts for metrics from the compliance checks table
    high_issues = len(checks_df[(checks_df['Priority'] == 'High') & (checks_df['Action'] == 'Remediate')])
    medium_issues = len(checks_df[(checks_df['Priority'] == 'Medium') & (checks_df['Action'] == 'Remediate')])
    low_issues = len(checks_df[(checks_df['Priority'] == 'Low') & (checks_df['Action'] == 'Remediate')])
    
    # Display metrics in a horizontal row
    st.markdown("""
    <style>
    /* Custom style for the Remediate button in the Optimization Checks table */
    div[data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stButton"] > button {
        background-color: #22c55e;
        color: white;
        border-color: #16a34a;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stButton"] > button:hover {
        background-color: #16a34a;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        create_metric_card(
            "High Priority Issues", 
            high_issues, 
            "Number of high priority issues to remediate.", 
            "metric-container-cost"
        )
    with col2:
        create_metric_card(
            "Medium Priority Issues", 
            medium_issues, 
            "Number of medium priority issues to remediate.", 
            "metric-container-accuracy-medium"
        )
    with col3:
        create_metric_card(
            "Low Priority Issues", 
            low_issues, 
            "Number of low priority issues to remediate.", 
            "metric-container-neutral"
        )
    
    st.markdown("---")

    st.subheader("Optimization Checks")

    # Add a toggle to show/hide N/A items
    show_na_items = st.toggle("Show N/A items", value=False)

    # Filter the DataFrame based on the toggle state
    if show_na_items:
        display_checks_df = checks_df
    else:
        display_checks_df = checks_df[checks_df['Action'] != 'N/A']

    # Header for the checks table
    header_cols = st.columns((5, 2, 2))
    header_cols[0].markdown("**Check**")
    header_cols[1].markdown("**Priority**")
    header_cols[2].markdown("**Action**")
    st.markdown("---")

    # Iterate over checks and display them as a custom table
    for index, row in display_checks_df.iterrows():
        check, priority, action = row['Check'], row['Priority'], row['Action']
        
        priority_color = '#ef4444' if priority == 'High' else '#facc15' if priority == 'Medium' else '#60a5fa'
        action_color = '#34d399' if action == 'Done' else '#ef4444' if action == 'Remediate' else '#60a5fa'

        row_cols = st.columns((5, 2, 2))
        
        row_cols[0].markdown(f'{check}', unsafe_allow_html=True)
        row_cols[1].markdown(f'<span style="color:{priority_color}">{priority}</span>', unsafe_allow_html=True)
        if action == 'Remediate':
            if row_cols[2].button("Remediate", key=f"remediate_{index}"):
                show_modal()
                # Update the DataFrame in session state
                st.session_state.checks_df.loc[index, 'Action'] = 'Done'
                st.rerun()
        else:
            row_cols[2].markdown(f'<span style="color:{action_color}">{action}</span>', unsafe_allow_html=True)

    # Display Model Efficiency Rankings
    if not model_recommendations.empty:
        st.subheader("Model Efficiency Rankings")
        
        # Convert to a format suitable for st.dataframe with colored bars
        model_efficiency_df = pd.DataFrame({
            "Model": model_recommendations['model'],
            "Cloud Platform": model_recommendations['cloud_platform'],
            "Department": model_recommendations['department'],
            "Project": model_recommendations['project'],
            "Environment": model_recommendations['environment'],
            "Release Version": model_recommendations['release_version'],
            "Efficiency Score": model_recommendations['efficiency_score'],
            "Latency (ms)": model_recommendations['latency_ms'],
            "Throughput (QPS)": model_recommendations['throughput_qps'],
            "Accuracy": model_recommendations['accuracy'],
            "Cost ($/1K tokens)": model_recommendations['cost_per_1k_tokens']
        })
        
        # Ensure Efficiency Score is float, between 0 and 1, and has no NaN
        model_efficiency_df["Efficiency Score"] = model_efficiency_df["Efficiency Score"].fillna(0).astype(float).clip(0, 1)
        
        st.dataframe(
            model_efficiency_df,
            column_config={
                "Model": "Model",
                "Cloud Platform": "Cloud Platform",
                "Department": "Department", 
                "Project": "Project",
                "Environment": "Environment",
                "Release Version": "Release Version",
                "Efficiency Score": st.column_config.ProgressColumn(
                    "Efficiency Score",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
                "Latency (ms)": st.column_config.NumberColumn(format="%.1f"),
                "Throughput (QPS)": st.column_config.NumberColumn(format="%.1f"),
                "Accuracy": st.column_config.NumberColumn(format="%.3f"),
                "Cost ($/1K tokens)": st.column_config.NumberColumn(format="$%.4f"),
            },
            hide_index=True,
        )

    # Display recommendations
    st.subheader("Automatic Recommendations")
    if recommendations:
        for rec in recommendations:
            st.write(rec['text'])
    else:
        st.write("No specific recommendations at this time.")
    
with tab5:
    st.header("Token Usage Analysis")

    if not model_df_filtered.empty:
        df = model_df_filtered.copy()

        # Add calculated columns for aggregation and charting
        df['daily_total_tokens'] = df['avg_tokens_per_minute'] * 60 * 24
        df['daily_token_cost'] = (df['daily_total_tokens'] / 1000) * df['cost_per_1k_tokens']

        # --- New Section: Token Limit Optimization Recommendations ---
        st.subheader("Token Limit Optimization Recommendations")

        # Aggregate data by model and department
        limit_analysis_df = df.groupby(['model', 'department']).agg(
            token_limit=('token_limit', 'first'),
            avg_max_tpm=('max_tokens_per_minute', 'mean'),
            total_exceptions=('token_limit_exception_count', 'sum')
        ).reset_index()

        if not limit_analysis_df.empty:
            limit_analysis_df['utilization_pct'] = (limit_analysis_df['avg_max_tpm'] / limit_analysis_df['token_limit']) * 100

            # Identify underutilized and overutilized pairs
            # Using a threshold > 5 to find frequent issues over the selected date range
            underutilized_pairs = limit_analysis_df[(limit_analysis_df['utilization_pct'] < 25) & (limit_analysis_df['total_exceptions'] == 0)]
            overutilized_pairs = limit_analysis_df[limit_analysis_df['total_exceptions'] > 100] 

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### Underutilized Limits (Potential Cost Savings)")
                if not underutilized_pairs.empty:
                    for _, row in underutilized_pairs.iterrows():
                        st.info(
                            f"**Model:** {row['model']} / **Dept:** {row['department']}\n"
                            f"- **Avg Peak TPM:** {row['avg_max_tpm']:,.0f}\n"
                            f"- **Current Limit:** {row['token_limit']:,.0f} (Utilization: {row['utilization_pct']:.1f}%)\n"
                            f"**Suggestion:** The token limit is significantly higher than peak usage. Consider lowering it to better match actual needs."
                        )
                else:
                    st.success("No significantly underutilized model/department pairs found.")

            with col2:
                st.markdown("##### Frequently Exceeded Limits (Performance Risk)")
                if not overutilized_pairs.empty:
                    for _, row in overutilized_pairs.iterrows():
                        st.warning(
                            f"**Model:** {row['model']} / **Dept:** {row['department']}\n"
                            f"- **Total Exceptions:** {row['total_exceptions']:,.0f} in selected period\n"
                            f"- **Current Limit:** {row['token_limit']:,.0f}\n"
                            f"**Suggestion:** The token limit is frequently exceeded. Consider increasing it to prevent service disruptions and improve response reliability."
                        )
                else:
                    st.success("No model/department pairs with frequent limit exceptions found.")


        st.subheader("Monthly Token Limit Exceptions")
        # Resample data for monthly exceptions
        df_monthly_exceptions = df.set_index('date').groupby('model')['token_limit_exception_count'].resample('M').sum().reset_index()
        # Using .to_period('M').to_timestamp() to make sure the date is the first of the month for clean plotting
        df_monthly_exceptions['month'] = df_monthly_exceptions['date'].dt.to_period('M').dt.to_timestamp()

        exceed_chart_type = st.radio(
            "Select Chart Type for Monthly Token Limit Exceptions",
            options=["Bar", "Line", "Area"], # Bar is a better default for monthly sums
            horizontal=True,
            key="token_exception_chart_type_tab5_3"
        )
        if exceed_chart_type == "Bar":
            fig_exceptions = px.bar(df_monthly_exceptions, x='month', y='token_limit_exception_count', color='model', barmode='group', title="Monthly Token Limit Exceptions")
        elif exceed_chart_type == "Line":
            fig_exceptions = px.line(df_monthly_exceptions, x='month', y='token_limit_exception_count', color='model', title="Monthly Token Limit Exceptions")
        elif exceed_chart_type == "Area":
            fig_exceptions = px.area(df_monthly_exceptions, x='month', y='token_limit_exception_count', color='model', title="Monthly Token Limit Exceptions")
        fig_exceptions.update_layout(xaxis_title="Month", yaxis_title="Total Exceptions")
        st.plotly_chart(fig_exceptions, use_container_width=True)


        st.subheader("Tokens per Minute Analysis")
        token_rate_chart_type = st.radio(
            "Select Chart Type for Tokens per Minute",
            options=["Line", "Bar", "Area"],
            horizontal=True,
            key="token_rate_chart_type_tab5_2"
        )
        token_rate_df = df.melt(
            id_vars=['date', 'model'], 
            value_vars=['avg_tokens_per_minute', 'max_tokens_per_minute'], 
            var_name='Metric', 
            value_name='Tokens per Minute'
        )
        token_rate_df['Metric'] = token_rate_df['Metric'].str.replace('_', ' ').str.title()
        if token_rate_chart_type == "Line":
            fig_token_rate = px.line(
                token_rate_df, 
                x='date', 
                y='Tokens per Minute', 
                color='model', 
                line_dash='Metric', 
                title="Tokens per Minute (Average vs Max)"
            )
        elif token_rate_chart_type == "Bar":
            fig_token_rate = px.bar(
                token_rate_df, 
                x='date', 
                y='Tokens per Minute', 
                color='model', 
                barmode='group', 
                title="Tokens per Minute (Average vs Max)"
            )
        elif token_rate_chart_type == "Area":
            fig_token_rate = px.area(
                token_rate_df, 
                x='date', 
                y='Tokens per Minute', 
                color='model', 
                line_group='Metric', 
                title="Tokens per Minute (Average vs Max)"
            )
        st.plotly_chart(fig_token_rate, use_container_width=True)
        
        st.subheader("Token Volume Over Time")
        token_chart_type = st.radio(
            "Select Chart Type for Daily Total Tokens",
            options=["Line", "Bar", "Area"],
            horizontal=True,
            key="token_total_chart_type_tab5_1"
        )
        if token_chart_type == "Line":
            fig_total_tokens = px.line(df, x='date', y='daily_total_tokens', color='model', title="Daily Total Tokens")
        elif token_chart_type == "Bar":
            fig_total_tokens = px.bar(df, x='date', y='daily_total_tokens', color='model', barmode='group', title="Daily Total Tokens")
        elif token_chart_type == "Area":
            fig_total_tokens = px.area(df, x='date', y='daily_total_tokens', color='model', title="Daily Total Tokens")
        st.plotly_chart(fig_total_tokens, use_container_width=True)

        st.subheader("Token Cost Analysis")
        token_cost_chart_type = st.radio(
            "Select Chart Type for Daily Token Cost",
            options=["Line", "Bar", "Area"],
            horizontal=True,
            key="token_cost_chart_type_tab5_4"
        )
        if token_cost_chart_type == "Line":
            fig_daily_cost = px.line(df, x='date', y='daily_token_cost', color='model', title="Daily Total Token Cost ($)")
        elif token_cost_chart_type == "Bar":
            fig_daily_cost = px.bar(df, x='date', y='daily_token_cost', color='model', barmode='group', title="Daily Total Token Cost ($)")
        elif token_cost_chart_type == "Area":
            fig_daily_cost = px.area(df, x='date', y='daily_token_cost', color='model', title="Daily Total Token Cost ($)")
        fig_daily_cost.update_layout(yaxis_title="Cost ($)")
        st.plotly_chart(fig_daily_cost, use_container_width=True)

        st.subheader("Cumulative Token Cost Analysis")
        cumulative_cost_chart_type = st.radio(
            "Select Chart Type for Cumulative Token Cost",
            options=["Line", "Bar", "Area"],
            horizontal=True,
            key="token_cumulative_chart_type_tab5_5"
        )
        if cumulative_cost_chart_type == "Line":
            df['cumulative_cost'] = df.sort_values('date').groupby('model')['daily_token_cost'].cumsum()
            fig_cumulative_cost = px.line(df, x='date', y='cumulative_cost', color='model', title="Cumulative Token Cost per Model ($)")
        elif cumulative_cost_chart_type == "Bar":
            df['cumulative_cost'] = df.sort_values('date').groupby('model')['daily_token_cost'].cumsum()
            fig_cumulative_cost = px.bar(df, x='date', y='cumulative_cost', color='model', barmode='group', title="Cumulative Token Cost per Model ($)")
        elif cumulative_cost_chart_type == "Area":
            df['cumulative_cost'] = df.sort_values('date').groupby('model')['daily_token_cost'].cumsum()
            fig_cumulative_cost = px.area(df, x='date', y='cumulative_cost', color='model', title="Cumulative Token Cost per Model ($)")
        fig_cumulative_cost.update_layout(yaxis_title="Cumulative Cost ($)")
        st.plotly_chart(fig_cumulative_cost, use_container_width=True)

    else:
        st.warning("No data to display for the selected filters.")

with tab6:
    st.header("Configure Alerts")

    # Define all available metrics for alerting
    all_metrics = (
        model_df.columns.drop(['date', 'model', 'token_limit']).tolist() +
        infra_df.columns.drop(['date', 'component', 'error_rate']).tolist()
    )
    
    # Initialize session state
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'edit_idx' not in st.session_state:
        st.session_state.edit_idx = None

    # --- EDIT ALERT UI ---
    if st.session_state.edit_idx is not None:
        st.subheader(f"Edit Alert: {st.session_state.alerts[st.session_state.edit_idx].get('name', '')}")
        alert_to_edit = st.session_state.alerts[st.session_state.edit_idx]
        
        metric_options = sorted(list(set(all_metrics)))
        condition_options = ["Above", "Below", "Between", "Increases by %", "Decreases by %"]
        
        with st.form("edit_alert_form"):
            st.text_input("Alert Name", key="edit_name", value=alert_to_edit.get('name'))
            st.selectbox("Select Metric", options=metric_options, key="edit_metric", index=metric_options.index(alert_to_edit['metric']))
            st.selectbox("Condition", options=condition_options, key="edit_condition", index=condition_options.index(alert_to_edit['condition']))

            thresholds = alert_to_edit.get('thresholds', {})
            comparison_period = alert_to_edit.get('comparison_period')

            if st.session_state.edit_condition == "Between":
                st.session_state.edit_lower_bound = st.number_input("Lower Bound", value=thresholds.get('lower_bound'))
                st.session_state.edit_upper_bound = st.number_input("Upper Bound", value=thresholds.get('upper_bound'))
            elif st.session_state.edit_condition in ["Increases by %", "Decreases by %"]:
                st.session_state.edit_percentage = st.number_input("Percentage", value=thresholds.get('percentage'))
                period_options = ["Previous Value", "7-Day Average", "30-Day Average"]
                st.selectbox(
                    "Compare Against", 
                    options=period_options, 
                    key="edit_comparison_period",
                    index=period_options.index(comparison_period) if comparison_period in period_options else 0
                )
            else:
                st.session_state.edit_value = st.number_input("Threshold Value", value=thresholds.get('value'))

            st.text_input("Email for Notification", key="edit_email", value=alert_to_edit.get('email'))

            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Update Alert"):
                    # Update logic here based on new inputs
                    updated_thresholds = {}
                    if st.session_state.edit_condition == "Between":
                        updated_thresholds = {'lower_bound': st.session_state.edit_lower_bound, 'upper_bound': st.session_state.edit_upper_bound}
                    elif st.session_state.edit_condition in ["Increases by %", "Decreases by %"]:
                        updated_thresholds = {'percentage': st.session_state.edit_percentage}
                    else:
                        updated_thresholds = {'value': st.session_state.edit_value}

                    st.session_state.alerts[st.session_state.edit_idx] = {
                        "name": st.session_state.edit_name,
                        "metric": st.session_state.edit_metric,
                        "condition": st.session_state.edit_condition,
                        "thresholds": updated_thresholds,
                        "email": st.session_state.edit_email,
                        "comparison_period": st.session_state.get('edit_comparison_period') if st.session_state.edit_condition in ["Increases by %", "Decreases by %"] else None
                    }
                    st.session_state.edit_idx = None
                    st.success("Alert updated successfully!")
                    st.rerun()
            with col2:
                if st.form_submit_button("Cancel"):
                    st.session_state.edit_idx = None
                    st.rerun()
    else:
        # --- CREATE NEW ALERT UI ---
        st.subheader("Set a New Alert")

        # After an alert is set, show success and reset widgets
        if st.session_state.get("alert_just_set"):
            st.success(f"Alert set for '{st.session_state.last_alert_metric}'.")
            st.session_state.alert_metric = None
            st.session_state.alert_condition = None
            st.session_state.alert_just_set = False
            if "last_alert_metric" in st.session_state:
                del st.session_state.last_alert_metric

        metric_to_alert = st.selectbox(
            "Select Metric",
            options=sorted(list(set(all_metrics))),
            key="alert_metric",
            placeholder="Select a metric...",
            index=None
        )
        
        condition_type = st.selectbox(
            "Condition", 
            options=["Above", "Below", "Between", "Increases by %", "Decreases by %"], 
            key="alert_condition",
            placeholder="Select a condition...",
            index=None
        )

        with st.form("create_alert_form", clear_on_submit=True):
            alert_name = st.text_input("Alert Name", key="alert_name_val", placeholder="e.g., High CPU on Model Server")
            
            threshold_inputs = {}
            comparison_period = None
            current_condition = st.session_state.get('alert_condition')

            if current_condition:
                t_col1, t_col2 = st.columns(2)
                with t_col1:
                    if current_condition in ["Above", "Below"]:
                        threshold_inputs['value'] = st.number_input("Threshold Value", step=1.0, key="threshold_val")
                    elif "by %" in current_condition:
                        threshold_inputs['percentage'] = st.number_input("Percentage", min_value=0.0, max_value=100.0, step=1.0, key="percentage_val")
                        comparison_period = st.selectbox(
                            "Compare Against", 
                            options=["Previous Value", "7-Day Average", "30-Day Average"], 
                            key="comparison_period_val"
                        )
                    elif current_condition == "Between":
                        threshold_inputs['lower_bound'] = st.number_input("Lower Bound", step=1.0, key="lower_bound_val")
                
                with t_col2:
                    if current_condition == "Between":
                        threshold_inputs['upper_bound'] = st.number_input("Upper Bound", step=1.0, key="upper_bound_val")

            email_to_notify = st.text_input("Email for Notification", key="email_val")
            
            submitted = st.form_submit_button("Set Alert")
            
            if submitted:
                current_metric = st.session_state.get('alert_metric')
                
                # --- Validation ---
                error = False
                if not all([current_metric, current_condition, alert_name]):
                    st.error("Please provide an alert name, metric, and condition.")
                    error = True
                
                if not error:
                    new_alert = {
                        "name": alert_name,
                        "metric": current_metric,
                        "condition": current_condition,
                        "thresholds": threshold_inputs,
                        "email": email_to_notify,
                        "comparison_period": comparison_period
                    }
                    st.session_state.alerts.append(new_alert)
                    st.session_state.last_alert_metric = current_metric
                    st.session_state.alert_just_set = True
                    st.rerun()

        st.divider()

        # --- Display Active Alerts ---
        st.subheader("Active Alerts")

        def format_alert_text(alert):
            metric = f"**{alert['metric']}**"
            condition = alert['condition']
            thresholds = alert['thresholds']
            email = f"**{alert['email']}**"
            comparison_period = alert.get('comparison_period', 'Previous Value')
            
            if condition == "Above":
                return f"Monitor {metric} to not go above **{thresholds['value']}**. Notify: {email}"
            elif condition == "Below":
                return f"Monitor {metric} to not go below **{thresholds['value']}**. Notify: {email}"
            elif condition == "Between":
                return f"Monitor {metric} to be between **{thresholds['lower_bound']}** and **{thresholds['upper_bound']}**. Notify: {email}"
            elif condition == "Increases by %":
                return f"Monitor {metric} for an increase of more than **{thresholds['percentage']}%** compared to the **{comparison_period.lower()}**. Notify: {email}"
            elif condition == "Decreases by %":
                return f"Monitor {metric} for a decrease of more than **{thresholds['percentage']}%** compared to the **{comparison_period.lower()}**. Notify: {email}"
            return "Invalid alert configuration."

        if not st.session_state.alerts:
            st.info("No alerts configured yet.")
        else:
            for i, alert in enumerate(st.session_state.alerts):
                col1, col2, col3 = st.columns([8, 1, 1])
                with col1:
                    st.info(f"**{alert.get('name', f'Alert {i+1}')}:** {format_alert_text(alert)}")
                with col2:
                    if st.button("Edit", key=f"edit_{i}"):
                        st.session_state.edit_idx = i
                        st.rerun()
                with col3:
                    if st.button("Delete", key=f"delete_{i}"):
                        st.session_state.alerts.pop(i)
                        st.rerun()

    # --- Check for Triggered Alerts ---
    st.subheader("Triggered Alerts")
    
    triggered_alerts = {}  # Using a dictionary to group by metric

    for alert in st.session_state.alerts:
        metric = alert['metric']
        condition = alert['condition']
        thresholds = alert['thresholds']
        alert_name = alert.get('name', 'Unnamed Alert')
        comparison_period = alert.get('comparison_period', 'Previous Value')

        if metric not in triggered_alerts:
            triggered_alerts[metric] = []

        for df, entity_col in [(model_df_filtered, 'model'), (infra_df_filtered, 'component')]:
            if metric in df.columns:
                sorted_df = df.sort_values('date')
                
                if condition in ["Increases by %", "Decreases by %"]:
                    for entity_name, group in sorted_df.groupby(entity_col):
                        if len(group) >= 2:
                            
                            # Determine the previous value based on the comparison period
                            if comparison_period == "7-Day Average" and len(group) >= 8:
                                previous = group[metric].rolling(window=7, min_periods=1).mean().iloc[-2]
                            elif comparison_period == "30-Day Average" and len(group) >= 31:
                                previous = group[metric].rolling(window=30, min_periods=1).mean().iloc[-2]
                            else: # Default to "Previous Value"
                                previous = group[metric].iloc[-2]

                            latest = group[metric].iloc[-1]
                            
                            if previous is not None and previous != 0:
                                change = ((latest - previous) / abs(previous)) * 100
                                if condition == "Increases by %" and change > thresholds['percentage']:
                                    message = f"**{alert_name}:** {entity_col.title()} **{entity_name}**'s metric increased by **{change:.2f}%** compared to the {comparison_period.lower()}, exceeding **{thresholds['percentage']}%**."
                                    triggered_alerts[metric].append(message)
                                elif condition == "Decreases by %" and change < -thresholds['percentage']:
                                    message = f"**{alert_name}:** {entity_col.title()} **{entity_name}**'s metric decreased by **{abs(change):.2f}%** compared to the {comparison_period.lower()}, exceeding **{thresholds['percentage']}%**."
                                    triggered_alerts[metric].append(message)
                else:
                    latest_values = sorted_df.groupby(entity_col)[metric].last()
                    for entity_name, value in latest_values.items():
                        if condition == "Above" and 'value' in thresholds and value > thresholds['value']:
                            message = f"**{alert_name}:** {entity_col.title()} **{entity_name}**'s value of **{value:.2f}** is above the threshold of **{thresholds['value']}**."
                            triggered_alerts[metric].append(message)
                        elif condition == "Below" and 'value' in thresholds and value < thresholds['value']:
                            message = f"**{alert_name}:** {entity_col.title()} **{entity_name}**'s value of **{value:.2f}** is below the threshold of **{thresholds['value']}**."
                            triggered_alerts[metric].append(message)
                        elif condition == "Between" and 'lower_bound' in thresholds and 'upper_bound' in thresholds and thresholds['lower_bound'] < value < thresholds['upper_bound']:
                            message = f"**{alert_name}:** {entity_col.title()} **{entity_name}**'s value of **{value:.2f}** is between **{thresholds['lower_bound']}** and **{thresholds['upper_bound']}**."
                            triggered_alerts[metric].append(message)

    # Filter out metrics with no triggered alerts and display the rest
    final_triggered_alerts = {k: v for k, v in triggered_alerts.items() if v}

    if not final_triggered_alerts:
        st.success("All systems normal. No alerts triggered based on the latest data.")
    else:
        for metric, messages in final_triggered_alerts.items():
            with st.expander(f"Metric: {metric.replace('_', ' ').title()}", expanded=True):
                for message in messages:
                    st.warning(message)

with tab7:
    st.header("Billing Details")

    # Load data from the static file (or generate it if it doesn't exist)
    with st.spinner("Loading billing data..."):
        try:
            billing_df = mock_billing_service.load_or_generate_billing_data()
        except Exception as e:
            st.error(f"Failed to load or generate billing data: {e}")
            billing_df = pd.DataFrame()

    if billing_df.empty:
        st.warning("No billing data found. Try regenerating the data.")
    else:
        # Ensure Date column is datetime for filtering
        billing_df['Date'] = pd.to_datetime(billing_df['Date'])

        # Filter the loaded data based on the sidebar controls
        start_date, end_date = date_range
        
        filtered_billing_df = billing_df[
            (billing_df['Date'].dt.date >= start_date) & 
            (billing_df['Date'].dt.date <= end_date)
        ]
        
        if selected_projects:
            filtered_billing_df = filtered_billing_df[filtered_billing_df['Project'].isin(selected_projects)]

        if filtered_billing_df.empty:
            st.warning("No billing data found for the selected filters.")
            st.stop()

        # Display metrics and charts using the filtered data
        st.subheader("Summary")
        total_cost = filtered_billing_df["Cost (USD)"].sum()
        total_tokens = filtered_billing_df[filtered_billing_df["Unit"] == "tokens"]["Usage"].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            create_metric_card("Total Billed Cost", f"${total_cost:,.2f}", "Total cost from the mock billing service for the selected period.", "metric-container-cost")
        with col2:
            create_metric_card("Total Tokens Billed", f"{total_tokens:,.0f}", "Total tokens consumed and billed across all services.", "metric-container-neutral")
        with col3:
            create_metric_card("Billing Records", f"{len(filtered_billing_df):,}", "Number of individual billing line items returned.", "metric-container-neutral")

        st.markdown("---")

        st.subheader("Cost Over Time")
        cost_over_time = filtered_billing_df.groupby("Date")["Cost (USD)"].sum().reset_index()
        fig_cost_time = px.area(cost_over_time, x="Date", y="Cost (USD)", title="Daily Billed Cost from Mock API")
        st.plotly_chart(fig_cost_time, use_container_width=True)

        st.subheader("Detailed Billing Records")
        st.dataframe(
            filtered_billing_df.sort_values(by=["Date", "Project", "Service"], ascending=[False, True, True]),
            use_container_width=True,
            hide_index=True
        )

# Add download functionality
st.sidebar.header("Export Data")

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

csv_model = convert_df_to_csv(model_df_filtered)
csv_infra = convert_df_to_csv(infra_df_filtered)

st.sidebar.download_button(
    label="Download Filtered Model Data as CSV",
    data=csv_model,
    file_name='genai_model_metrics_filtered.csv',
    mime='text/csv',
)

st.sidebar.download_button(
    label="Download Filtered Infrastructure Data as CSV",
    data=csv_infra,
    file_name='genai_infrastructure_metrics_filtered.csv',
    mime='text/csv',
)

st.sidebar.markdown("---")
st.sidebar.info("""
🏢 **Enterprise Demo**: This dashboard uses simulated data representing a multi-cloud GenAI deployment across AWS, GCP, Azure, Snowflake, and Databricks. 

In production, connect to your actual monitoring systems and cost management APIs.
""")

# Cognizant footer branding
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px;">
    <p style="color: #1f77b4; font-weight: bold; margin: 0;">Cognizant</p>
    <p style="color: #666; font-size: 12px; margin: 0;">AI & Analytics Solutions</p>
    <p style="color: #666; font-size: 10px; margin: 0;">© 2025 Cognizant Technology Solutions</p>
</div>
""", unsafe_allow_html=True)

# Main footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p style="color: #1f77b4; font-weight: bold;">Cognizant Prometheus Insights</p>
</div>
""", unsafe_allow_html=True)