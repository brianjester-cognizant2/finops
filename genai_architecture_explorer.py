import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="GenAI Architecture Explorer",
    page_icon="🧠",
    layout="wide"
)

# Title and introduction
st.title("GenAI Architecture Explorer")
st.markdown("""
This application helps you explore and analyze your GenAI architecture, 
identify optimization opportunities, and visualize key performance metrics.
""")

# Sidebar configuration
st.sidebar.header("Settings")

# Sample data generation function
def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
    
    # Model performance data
    models = ['GPT-4', 'Claude 3', 'Llama 3', 'Mixtral', 'PaLM 2']
    model_data = []
    
    for model in models:
        base_latency = np.random.uniform(100, 500)
        base_throughput = np.random.uniform(10, 50)
        base_accuracy = np.random.uniform(0.7, 0.95)
        base_cost = np.random.uniform(0.01, 0.1)
        
        for date in dates:
            # Add some trend and random noise
            trend_factor = 1 - (0.001 * (date - dates[0]).days)  # Gradual improvement over time
            random_factor = np.random.normal(1, 0.05)  # Daily variation
            
            latency = base_latency * trend_factor * random_factor
            throughput = base_throughput / trend_factor * random_factor
            accuracy = min(0.99, base_accuracy / trend_factor * random_factor)
            cost = base_cost * trend_factor * random_factor
            
            model_data.append({
                'date': date,
                'model': model,
                'latency_ms': latency,
                'throughput_qps': throughput,
                'accuracy': accuracy,
                'cost_per_1k_tokens': cost,
                'memory_usage_gb': np.random.uniform(4, 16),
                'gpu_utilization': np.random.uniform(0.4, 0.95)
            })
    
    # Infrastructure data
    infra_data = []
    components = ['API Gateway', 'Load Balancer', 'Model Server', 'Cache', 'Database']
    
    for component in components:
        base_cpu = np.random.uniform(20, 60)
        base_memory = np.random.uniform(30, 70)
        base_requests = np.random.uniform(100, 1000)
        
        for date in dates:
            # Weekly pattern + trend
            day_of_week = date.dayofweek
            weekly_factor = 1 + 0.2 * (day_of_week < 5)  # Higher on weekdays
            trend_factor = 1 + (0.0005 * (date - dates[0]).days)  # Gradual increase in load
            
            cpu_usage = base_cpu * weekly_factor * trend_factor * np.random.normal(1, 0.1)
            memory_usage = base_memory * weekly_factor * trend_factor * np.random.normal(1, 0.05)
            requests = base_requests * weekly_factor * trend_factor * np.random.normal(1, 0.2)
            
            infra_data.append({
                'date': date,
                'component': component,
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

# Load or generate data
if 'model_data' not in st.session_state or 'infra_data' not in st.session_state:
    with st.spinner('Generating sample data...'):
        st.session_state.model_data, st.session_state.infra_data = generate_sample_data()

model_df = st.session_state.model_data
infra_df = st.session_state.infra_data

# Calculate error rate on the main dataframe to ensure it's always available
infra_df['error_rate'] = infra_df['errors_per_minute'] / infra_df['requests_per_minute']

# Date filter for analysis
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(model_df['date'].min().date(), model_df['date'].max().date()),
    min_value=model_df['date'].min().date(),
    max_value=model_df['date'].max().date()
)

# Filter data based on selected date range
if len(date_range) == 2:
    start_date, end_date = date_range
    model_df_filtered = model_df[(model_df['date'].dt.date >= start_date) & (model_df['date'].dt.date <= end_date)]
    infra_df_filtered = infra_df[(infra_df['date'].dt.date >= start_date) & (infra_df['date'].dt.date <= end_date)]
else:
    model_df_filtered = model_df
    infra_df_filtered = infra_df

# Model selection
selected_models = st.sidebar.multiselect(
    "Select Models to Display",
    options=model_df['model'].unique(),
    default=model_df['model'].unique()[:3]  # Default select first 3 models
)

# Infrastructure component selection
selected_components = st.sidebar.multiselect(
    "Select Infrastructure Components",
    options=infra_df['component'].unique(),
    default=infra_df['component'].unique()
)

# Metric selection for models
model_metrics = st.sidebar.multiselect(
    "Select Model Metrics to Analyze",
    options=['latency_ms', 'throughput_qps', 'accuracy', 'cost_per_1k_tokens', 'memory_usage_gb', 'gpu_utilization'],
    default=['latency_ms', 'accuracy', 'cost_per_1k_tokens']
)

# Filter data based on selections
model_df_filtered = model_df_filtered[model_df_filtered['model'].isin(selected_models)]
infra_df_filtered = infra_df_filtered[infra_df_filtered['component'].isin(selected_components)]

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Model Performance", "Infrastructure Metrics", "Architecture Overview", "Optimization Recommendations", "Alerts"])

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
    
    cols = st.columns(len(model_metrics) if len(model_metrics) > 0 else 1)
    
    for i, metric in enumerate(model_metrics):
        with cols[i % len(cols)]:
            pretty_metric = metric.replace('_', ' ').title()
            st.write(f"**{pretty_metric}**")
            
            fig = px.line(
                model_df_filtered, 
                x='date', 
                y=metric, 
                color='model',
                title=f"{pretty_metric} Over Time"
            )
            
            # Add a trend line for each model
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
    
    # Create line chart for the selected metric
    fig = px.line(
        infra_df_filtered, 
        x='date', 
        y=selected_infra_metric, 
        color='component',
        title=f"{selected_infra_metric.replace('_', ' ').title()} Over Time"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap between metrics
    st.subheader("Correlation Between Infrastructure Metrics")
    
    # Calculate average metrics by component
    avg_metrics_by_component = infra_df_filtered.groupby('component')[metric_options].mean().reset_index()
    
    # Create heatmap
    if not avg_metrics_by_component.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = avg_metrics_by_component[metric_options].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        plt.title('Correlation Between Infrastructure Metrics')
        st.pyplot(fig)
    
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
    
    # Calculate average resource usage by component
    avg_resources = infra_df_filtered.groupby('component')[['cpu_usage_percent', 'memory_usage_percent']].mean().reset_index()
    
    # Create a horizontal bar chart for resource allocation
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
    
    # Calculate model efficiency scores
    latest_metrics = model_df_filtered[model_df_filtered['date'] == model_df_filtered['date'].max()].copy()
    
    if not latest_metrics.empty:
        # Normalize metrics for scoring
        latest_metrics['latency_norm'] = 1 - (latest_metrics['latency_ms'] - latest_metrics['latency_ms'].min()) / (latest_metrics['latency_ms'].max() - latest_metrics['latency_ms'].min())
        latest_metrics['throughput_norm'] = (latest_metrics['throughput_qps'] - latest_metrics['throughput_qps'].min()) / (latest_metrics['throughput_qps'].max() - latest_metrics['throughput_qps'].min())
        latest_metrics['accuracy_norm'] = (latest_metrics['accuracy'] - latest_metrics['accuracy'].min()) / (latest_metrics['accuracy'].max() - latest_metrics['accuracy'].min())
        latest_metrics['cost_norm'] = 1 - (latest_metrics['cost_per_1k_tokens'] - latest_metrics['cost_per_1k_tokens'].min()) / (latest_metrics['cost_per_1k_tokens'].max() - latest_metrics['cost_per_1k_tokens'].min())
        latest_metrics['memory_norm'] = 1 - (latest_metrics['memory_usage_gb'] - latest_metrics['memory_usage_gb'].min()) / (latest_metrics['memory_usage_gb'].max() - latest_metrics['memory_usage_gb'].min())
        
        # Calculate efficiency score (higher is better)
        latest_metrics['efficiency_score'] = (
            latest_metrics['latency_norm'] * 0.25 +
            latest_metrics['throughput_norm'] * 0.2 +
            latest_metrics['accuracy_norm'] * 0.3 +
            latest_metrics['cost_norm'] * 0.15 +
            latest_metrics['memory_norm'] * 0.1
        )
        
        # Sort by efficiency score
        model_recommendations = latest_metrics[['model', 'efficiency_score', 'latency_ms', 'throughput_qps', 'accuracy', 'cost_per_1k_tokens']].sort_values('efficiency_score', ascending=False)
        
        st.subheader("Model Efficiency Rankings")
        
        # Convert to a format suitable for st.dataframe with colored bars
        model_efficiency_df = pd.DataFrame({
            "Model": model_recommendations['model'],
            "Efficiency Score": model_recommendations['efficiency_score'],
            "Latency (ms)": model_recommendations['latency_ms'],
            "Throughput (QPS)": model_recommendations['throughput_qps'],
            "Accuracy": model_recommendations['accuracy'],
            "Cost ($/1K tokens)": model_recommendations['cost_per_1k_tokens']
        })
        
        st.dataframe(
            model_efficiency_df,
            column_config={
                "Model": "Model",
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
    
    # Generate recommendations based on data analysis
    st.subheader("Automatic Recommendations")
    
    recommendations = []
    
    # Check for model-specific recommendations
    if not latest_metrics.empty:
        # Find most efficient model
        best_model = model_recommendations.iloc[0]['model']
        recommendations.append(f"🔹 Consider using **{best_model}** as your primary model based on overall efficiency score.")
        
        # Check for cost optimization
        cost_efficient = model_recommendations.sort_values('cost_per_1k_tokens').iloc[0]['model']
        if cost_efficient != best_model:
            recommendations.append(f"🔹 For cost-sensitive applications, **{cost_efficient}** provides the best value.")
        
        # Check for latency optimization
        low_latency = model_recommendations.sort_values('latency_ms').iloc[0]['model']
        if low_latency != best_model:
            recommendations.append(f"🔹 For latency-critical applications, **{low_latency}** provides the fastest response times.")
    
    # Check for infrastructure recommendations
    if not bottlenecks.empty:
        # CPU bottlenecks
        cpu_bottlenecks = bottlenecks[bottlenecks['cpu_max'] > 80]['component'].tolist()
        if cpu_bottlenecks:
            recommendations.append(f"🔹 Consider scaling up or out the following components with high CPU usage: **{', '.join(cpu_bottlenecks)}**.")
        
        # Memory bottlenecks
        memory_bottlenecks = bottlenecks[bottlenecks['memory_max'] > 80]['component'].tolist()
        if memory_bottlenecks:
            recommendations.append(f"🔹 Increase memory allocation for: **{', '.join(memory_bottlenecks)}**.")
        
        # Response time bottlenecks
        response_bottlenecks = bottlenecks[bottlenecks['response_max'] > 300]['component'].tolist()
        if response_bottlenecks:
            recommendations.append(f"🔹 Optimize or scale the following components to reduce response times: **{', '.join(response_bottlenecks)}**.")
    
    # Error rate recommendations
    high_error_components = infra_df_filtered.groupby('component')['error_rate'].mean()
    high_error_components = high_error_components[high_error_components > 0.01].index.tolist()
    if high_error_components:
        recommendations.append(f"🔹 Investigate and reduce error rates in: **{', '.join(high_error_components)}**.")
    
    # Cache optimization recommendations
    cache_data = infra_df_filtered[infra_df_filtered['component'] == 'Cache']
    if not cache_data.empty and cache_data['cpu_usage_percent'].mean() > 60:
        recommendations.append("🔹 Consider implementing a more efficient caching strategy or scaling your cache layer.")
    
    # Architecture recommendations
    recommendations.append("🔹 Consider implementing a model router to dynamically select the optimal model based on request characteristics.")
    recommendations.append("🔹 Add redundancy to critical components to improve system reliability.")
    
    # Display recommendations
    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.write("No specific recommendations at this time.")
    
    # Cost optimization section
    st.subheader("Cost Optimization Analysis")
    
    # Calculate average cost per model and usage over time
    model_cost_data = model_df_filtered.groupby(['model', pd.Grouper(key='date', freq='W')])['cost_per_1k_tokens'].mean().reset_index()
    
    fig = px.line(
        model_cost_data, 
        x='date', 
        y='cost_per_1k_tokens', 
        color='model',
        title="Weekly Average Cost per 1K Tokens"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate potential savings
    if not latest_metrics.empty:
        current_models = latest_metrics['model'].unique()
        if len(current_models) > 1:
            avg_cost = latest_metrics['cost_per_1k_tokens'].mean()
            min_cost = latest_metrics['cost_per_1k_tokens'].min()
            potential_savings_pct = ((avg_cost - min_cost) / avg_cost) * 100
            
            st.metric(
                label="Potential Cost Savings by Optimizing Model Selection",
                value=f"{potential_savings_pct:.1f}%",
                delta=f"${avg_cost - min_cost:.4f} per 1K tokens"
            )

with tab5:
    st.header("Configure and View Alerts")

    # Define all available metrics for alerting
    all_metrics = (
        model_df.columns.drop(['date', 'model']).tolist() +
        infra_df.columns.drop(['date', 'component', 'error_rate']).tolist()
    )
    
    # --- Alert Configuration UI ---
    st.subheader("Set a New Alert")
    
    # Initialize session state for alerts if it doesn't exist
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []

    with st.form("alert_form"):
        col1, col2, col3, col4 = st.columns([3, 2, 3, 1])
        
        with col1:
            metric_to_alert = st.selectbox("Select Metric", options=sorted(list(set(all_metrics))))
        with col2:
            threshold = st.number_input("Threshold Value", value=0.0)
        with col3:
            email_to_alert = st.text_input("Email for Notification", placeholder="example@domain.com")
        with col4:
            submitted = st.form_submit_button("Set Alert")

        if submitted:
            if email_to_alert and metric_to_alert:
                st.session_state.alerts.append({
                    "metric": metric_to_alert,
                    "threshold": threshold,
                    "email": email_to_alert
                })
                st.success(f"Alert set for '{metric_to_alert}' with threshold {threshold}.")
            else:
                st.error("Please provide a metric and an email address.")

    # --- Display Active Alerts ---
    st.subheader("Active Alerts")
    if not st.session_state.alerts:
        st.info("No alerts configured yet.")
    else:
        for i, alert in enumerate(st.session_state.alerts):
            st.info(f"**Alert {i+1}:** Monitor **{alert['metric']}** to not exceed **{alert['threshold']}**. Notify: **{alert['email']}**")

    # --- Check for Triggered Alerts ---
    st.subheader("Triggered Alerts")
    
    triggered_alerts_found = False
    
    for alert in st.session_state.alerts:
        metric = alert['metric']
        threshold = alert['threshold']
        
        # Check if the metric is from model data or infra data
        if metric in model_df_filtered.columns:
            latest_values = model_df_filtered.groupby('model')[metric].last()
            for model, value in latest_values.items():
                if value > threshold:
                    st.warning(f"**Alert Triggered:** Model **{model}**'s **{metric}** is **{value:.2f}**, which is above the threshold of **{threshold}**.")
                    triggered_alerts_found = True
        elif metric in infra_df_filtered.columns:
            latest_values = infra_df_filtered.groupby('component')[metric].last()
            for component, value in latest_values.items():
                if value > threshold:
                    st.warning(f"**Alert Triggered:** Component **{component}**'s **{metric}** is **{value:.2f}**, which is above the threshold of **{threshold}**.")
                    triggered_alerts_found = True

    if not triggered_alerts_found:
        st.success("All systems normal. No alerts triggered based on the latest data.")

# Add download functionality
st.sidebar.header("Export Data")

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

csv_model = convert_df_to_csv(model_df)
csv_infra = convert_df_to_csv(infra_df)

st.sidebar.download_button(
    label="Download Model Data as CSV",
    data=csv_model,
    file_name='genai_model_metrics.csv',
    mime='text/csv',
)

st.sidebar.download_button(
    label="Download Infrastructure Data as CSV",
    data=csv_infra,
    file_name='genai_infrastructure_metrics.csv',
    mime='text/csv',
)

# Footer with usage instructions
st.sidebar.markdown("---")
st.sidebar.markdown("""
### How to Use
1. Select date range for analysis
2. Choose models and components to compare
3. Navigate through tabs to explore different aspects
4. Export data for further analysis
""")

st.sidebar.markdown("---")
st.sidebar.info("This app uses sample data for demonstration purposes. In a real deployment, connect to your actual monitoring systems.")