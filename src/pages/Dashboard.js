import React from 'react';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import MetricCard from '../components/MetricCard';
import DataTable from '../components/DataTable';
import { summaryMetrics, costBreakdown, recentAnomalies, spendOverTime } from '../data/mockData';
import './Dashboard.css';

const COLORS = ['#a855f7', '#34d399', '#f97316', '#8b5cf6', '#84cc16', '#facc15'];

// Prepare data for charts
const parsedCostBreakdown = costBreakdown.map(item => ({
  ...item,
  spendValue: parseFloat(item.spend.replace(/[^0-9.-]+/g, "")),
}));

const spendByProject = Object.values(parsedCostBreakdown.reduce((acc, curr) => {
  if (!acc[curr.project]) {
    acc[curr.project] = { project: curr.project, spendValue: 0 };
  }
  acc[curr.project].spendValue += curr.spendValue;
  return acc;
}, {}));

function Dashboard() {
  return (
    <div className="dashboard">
      <section className="metrics-grid">
        {summaryMetrics.map((metric) => (
          <MetricCard key={metric.title} title={metric.title} value={metric.value} change={metric.change} />
        ))}
      </section>

      <section className="dashboard-section">
        <h2>Spend Overview (Month-to-Date)</h2>
        <div className="chart-wrapper">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={spendOverTime} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis tickFormatter={(value) => `$${value / 1000}k`} />
              <Tooltip formatter={(value) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value)} />
              <Legend />
              <Line type="monotone" dataKey="spend" stroke="#a855f7" strokeWidth={2} activeDot={{ r: 8 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <div className="charts-grid">
        <section className="dashboard-section">
          <h2>Cost Breakdown by Service</h2>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={parsedCostBreakdown}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="spendValue"
                  nameKey="service"
                >
                  {parsedCostBreakdown.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value)} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="dashboard-section">
          <h2>Total Spend by Project</h2>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={spendByProject} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="project" />
                <YAxis tickFormatter={(value) => `$${value / 1000}k`} />
                <Tooltip formatter={(value) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value)} />
                <Legend />
                <Bar dataKey="spendValue" name="Spend" fill="#34d399" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      <section className="dashboard-section">
        <h2>Cost Breakdown Details</h2>
        <DataTable
          headers={['Service', 'Project', 'Spend', 'Usage (Tokens)', 'Trend']}
          dataKeys={['service', 'project', 'spend', 'usage', 'trend']}
          rows={costBreakdown}
        />
      </section>

      <section className="dashboard-section">
        <h2>Recent Anomalies & Alerts</h2>
        <DataTable
          headers={['Timestamp', 'Service', 'Description', 'Severity']}
          dataKeys={['timestamp', 'service', 'description', 'severity']}
          rows={recentAnomalies}
        />
      </section>
    </div>
  );
}

export default Dashboard;