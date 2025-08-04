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
    dates = pd.date_range(start='2023-01-01', end= pd.Timestamp.today().strftime('%Y-%m-%d'), freq='D')
    
    # Model performance data
    models = {
        'GPT-4': {'token_limit': 80000, 'avg_tokens_per_request': 2500},
        'Claude 3': {'token_limit': 150000, 'avg_tokens_per_request': 3000},
        'Llama 3': {'token_limit': 70000, 'avg_tokens_per_request': 2000},
        'Mixtral': {'token_limit': 30000, 'avg_tokens_per_request': 1800},
        'PaLM 2': {'token_limit': 30000, 'avg_tokens_per_request': 1500}
    }
    model_data = []
    
    for model, properties in models.items():
        base_latency = np.random.uniform(100, 500)
        base_throughput = np.random.uniform(100, 5000)
        base_accuracy = np.random.uniform(0.7, 0.95)
        base_cost = np.random.uniform(0.01, 0.1)
        
        for date in dates:
            # Add some trend and random noise
            trend_factor = 1 - (0.0001 * (date - dates[0]).days)  # Gradual improvement over time
            random_factor = np.random.normal(1, 0.05)  # Daily variation
            
            latency = base_latency * trend_factor * random_factor
            throughput = base_throughput / trend_factor * random_factor
            accuracy = min(0.99, base_accuracy / trend_factor * random_factor)
            cost = base_cost * trend_factor * random_factor
            
            # Token usage simulation
            avg_tokens_per_minute = ( throughput / ( 60 * 24 ) )* properties['avg_tokens_per_request'] * np.random.normal(1, 0.15)
            max_tokens_per_minute = avg_tokens_per_minute * np.random.uniform(1.5, 7.5)
            minute_samples = np.random.normal(max_tokens_per_minute, max_tokens_per_minute * 0.4)
            exceedances = np.sum(minute_samples > properties['token_limit'])
            
            model_data.append({
                'date': date,
                'model': model,
                'latency_ms': latency,
                'throughput_qps': throughput,
                'accuracy': accuracy,
                'cost_per_1k_tokens': cost,
                'memory_usage_gb': np.random.uniform(4, 16),
                'gpu_utilization': np.random.uniform(0.4, 0.95),
                'avg_tokens_per_minute': avg_tokens_per_minute,
                'max_tokens_per_minute': max_tokens_per_minute,
                'token_limit_exceeded_count': exceedances,
                'token_limit': properties['token_limit']
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Model Performance", "Infrastructure Metrics", "Architecture Overview", "Optimization Recommendations", "Token Usage Analysis", "Alerts"])

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
        
        # Sort by efficiency score
        model_recommendations = latest_metrics[['model', 'efficiency_score', 'latency_ms', 'throughput_qps', 'accuracy', 'cost_per_1k_tokens']].sort_values('efficiency_score', ascending=False)

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

    # Create the compliance checks dataframe first to calculate metrics from it
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
    implemented = np.random.choice(['No', 'Yes', 'N/A'], size=len(checks_list))
    
    checks_df = pd.DataFrame({
        'Check': checks_list,
        'Priority': priorities,
        'Implemented': implemented
    })

    # Calculate counts for metrics from the compliance checks table
    high_issues = len(checks_df[(checks_df['Priority'] == 'High') & (checks_df['Implemented'] == 'No')])
    medium_issues = len(checks_df[(checks_df['Priority'] == 'Medium') & (checks_df['Implemented'] == 'No')])
    low_issues = len(checks_df[(checks_df['Priority'] == 'Low') & (checks_df['Implemented'] == 'No')])
    # Display metrics in a horizontal row
    st.markdown("""
    <style>
    .metric-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #262730;
        text-align: center;
        border: 1px solid #374151;
    }
    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        font-weight: 500;
        color: #9ca3af;
    }
    .metric-card p {
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
    }
    .metric-card.red p { color: #ef4444; }
    .metric-card.yellow p { color: #facc15; }
    .metric-card.blue p { color: #60a5fa; }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card red">
            <h3>High Priority</h3>
            <p>{high_issues}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card yellow">
            <h3>Medium Priority</h3>
            <p>{medium_issues}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card blue">
            <h3>Low Priority</h3>
            <p>{low_issues}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

    st.subheader("Optimization Checks")

    # Header for the checks table
    header_cols = st.columns((5, 2, 2))
    header_cols[0].markdown("**Check**")
    header_cols[1].markdown("**Priority**")
    header_cols[2].markdown("**Implemented**")
    st.markdown("---")

    # Iterate over checks and display them as a custom table
    for index, row in checks_df.iterrows():
        check, priority, implemented = row['Check'], row['Priority'], row['Implemented']
        
        priority_color = '#ef4444' if priority == 'High' else '#facc15' if priority == 'Medium' else '#60a5fa'
        impl_color = '#34d399' if implemented == 'Yes' else '#ef4444' if implemented == 'No' else '#60a5fa'

        row_cols = st.columns((5, 2, 2))
        
        check_text = f"{check} <a href='#' style='color: #60a5fa; text-decoration: none;'>(Remediate)</a>" if implemented == 'No' else check

        row_cols[0].markdown(f'<span style="color:{impl_color}">{check_text}</span>', unsafe_allow_html=True)
        row_cols[1].markdown(f'<span style="color:{priority_color}">{priority}</span>', unsafe_allow_html=True)
        row_cols[2].markdown(f'<span style="color:{impl_color}">{implemented}</span>', unsafe_allow_html=True)

    # Display Model Efficiency Rankings
    if not model_recommendations.empty:
        st.subheader("Model Efficiency Rankings")
        
        # test Convert to a format suitable for st.dataframe with colored bars
        model_efficiency_df = pd.DataFrame({
            "Model": model_recommendations['model'],
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

        # Add calculated columns
        df['daily_total_tokens'] = df['avg_tokens_per_minute'] * 60 * 24
        df['daily_token_cost'] = (df['daily_total_tokens'] / 1000) * df['cost_per_1k_tokens']

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
        # Add threshold lines
        model_limits = df.groupby('model')['token_limit'].first()
        model_colors = {trace.name: trace.line.color for trace in fig_token_rate.data}
        for model_name, limit in model_limits.items():
            if model_name in model_colors:
                fig_token_rate.add_hline(
                    y=limit,
                    line_dash="dot",
                    annotation_text=f"{model_name} Limit",
                    annotation_position="bottom right",
                    line_color=model_colors[model_name]
                )
        st.plotly_chart(fig_token_rate, use_container_width=True)

        st.subheader("Token Limit Exceedances")
        exceed_chart_type = st.radio(
            "Select Chart Type for Token Limit Exceedances",
            options=["Line", "Bar", "Area"],
            horizontal=True,
            key="token_exceed_chart_type_tab5_3"
        )
        if exceed_chart_type == "Line":
            fig_exceedances = px.line(df, x='date', y='token_limit_exceeded_count', color='model', title="Daily Token Limit Exceedance Count")
        elif exceed_chart_type == "Bar":
            fig_exceedances = px.bar(df, x='date', y='token_limit_exceeded_count', color='model', barmode='group', title="Daily Token Limit Exceedance Count")
        elif exceed_chart_type == "Area":
            fig_exceedances = px.area(df, x='date', y='token_limit_exceeded_count', color='model', title="Daily Token Limit Exceedance Count")
        st.plotly_chart(fig_exceedances, use_container_width=True)

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

        st.subheader("Token Limit Utilization")
        utilization_chart_type = st.radio(
            "Select Chart Type for Token Limit Utilization",
            options=["Bar", "Line", "Area"],
            horizontal=True,
            key="token_utilization_chart_type_tab5_6"
        )
        if 'token_limit' in df.columns:
            utilization_df = df.groupby('model').agg(
                avg_max_tokens=('max_tokens_per_minute', 'mean'),
                token_limit=('token_limit', 'first')
            ).reset_index()

            utilization_df['utilization_pct'] = (utilization_df['avg_max_tokens'] / utilization_df['token_limit']) * 100

            underutilized_models = utilization_df[utilization_df['utilization_pct'] < 25]

            if not underutilized_models.empty:
                st.warning("Found models with low token limit utilization:")
                for index, row in underutilized_models.iterrows():
                    st.markdown(f"- **{row['model']}**: Average peak usage is **{row['avg_max_tokens']:.0f} tokens/min**, which is only **{row['utilization_pct']:.1f}%** of its **{row['token_limit']:,}** token limit. Consider lowering the limit to better match actual usage.")
            else:
                st.success("All models show healthy token limit utilization based on peak usage.")

            if utilization_chart_type == "Bar":
                fig_utilization = px.bar(
                    utilization_df.sort_values('utilization_pct', ascending=False),
                    x='model',
                    y='utilization_pct',
                    title='Average Peak Token Usage vs. Token Limit',
                    labels={'utilization_pct': 'Utilization of Token Limit (%)', 'model': 'Model'},
                    color='utilization_pct',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
            elif utilization_chart_type == "Line":
                fig_utilization = px.line(
                    utilization_df.sort_values('utilization_pct', ascending=False),
                    x='model',
                    y='utilization_pct',
                    title='Average Peak Token Usage vs. Token Limit',
                    labels={'utilization_pct': 'Utilization of Token Limit (%)', 'model': 'Model'},
                    color='utilization_pct',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
            elif utilization_chart_type == "Area":
                fig_utilization = px.area(
                    utilization_df.sort_values('utilization_pct', ascending=False),
                    x='model',
                    y='utilization_pct',
                    title='Average Peak Token Usage vs. Token Limit',
                    labels={'utilization_pct': 'Utilization of Token Limit (%)', 'model': 'Model'},
                    color='utilization_pct',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
            fig_utilization.add_hline(y=25, line_dash="dash", annotation_text="Low Utilization Threshold (25%)", annotation_position="bottom right")
            st.plotly_chart(fig_utilization, use_container_width=True)
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