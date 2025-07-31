import React from 'react';
import MetricCard from '../components/MetricCard';
import CostChart from '../components/CostChart';
import DataTable from '../components/DataTable';
import { summaryMetrics, costBreakdown, recentAnomalies } from '../data/mockData';
import './Dashboard.css';

function Dashboard() {
  return (
    <div className="dashboard">
      <section className="metrics-grid">
        {summaryMetrics.map((metric, index) => (
          <MetricCard key={index} title={metric.title} value={metric.value} change={metric.change} />
        ))}
      </section>

      <section className="dashboard-section">
        <h2>Spend Overview</h2>
        <div className="chart-container">
          <CostChart />
        </div>
      </section>

      <section className="dashboard-section">
        <h2>Cost Breakdown by Service</h2>
        <DataTable
          headers={['Service', 'Project', 'Spend', 'Usage (Tokens)', 'Trend']}
          rows={costBreakdown}
        />
      </section>

      <section className="dashboard-section">
        <h2>Recent Anomalies & Alerts</h2>
        <DataTable
          headers={['Timestamp', 'Service', 'Description', 'Severity']}
          rows={recentAnomalies}
        />
      </section>
    </div>
  );
}

export default Dashboard;
