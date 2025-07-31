import React, { useState, useMemo } from 'react';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import MetricCard from '../components/MetricCard';
import DataTable from '../components/DataTable';
import { summaryMetrics, costBreakdown, recentAnomalies, spendOverTime } from '../data/mockData';
import './Dashboard.css';

const COLORS = ['#a855f7', '#34d399', '#f97316', '#8b5cf6', '#84cc16', '#facc15'];

function Dashboard() {
  const [selectedProject, setSelectedProject] = useState('All Projects');

  const projectNames = useMemo(() => ['All Projects', ...new Set(costBreakdown.map(item => item.project))], []);

  const {
    filteredCostBreakdown,
    filteredAnomalies,
    dynamicSummaryMetrics,
    parsedCostBreakdown,
    spendByProject,
    filteredSpendOverTime
  } = useMemo(() => {
    const fCostBreakdown = selectedProject === 'All Projects'
      ? costBreakdown
      : costBreakdown.filter(item => item.project === selectedProject);

    const serviceToProjectMap = costBreakdown.reduce((acc, item) => {
      acc[item.service] = item.project;
      return acc;
    }, {});
    const fAnomalies = selectedProject === 'All Projects'
      ? recentAnomalies
      : recentAnomalies.filter(anomaly => serviceToProjectMap[anomaly.service] === selectedProject);

    const pCostBreakdown = fCostBreakdown.map(item => ({
      ...item,
      spendValue: parseFloat(item.spend.replace(/[^0-9.-]+/g, "")),
    }));

    const totalSpend = pCostBreakdown.reduce((sum, item) => sum + item.spendValue, 0);
    const dSummaryMetrics = [
      { title: 'Total Spend (Month-to-Date)', value: new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(totalSpend), change: summaryMetrics[0].change },
      { title: 'Forecasted Spend', value: new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(totalSpend * 1.75), change: summaryMetrics[1].change },
      { title: 'Active Projects', value: selectedProject === 'All Projects' ? new Set(costBreakdown.map(p => p.project)).size : 1 },
      { ...summaryMetrics[3], value: selectedProject === 'All Projects' ? '4' : '1' },
    ];

    const sByProject = Object.values(pCostBreakdown.reduce((acc, curr) => {
      if (!acc[curr.project]) {
        acc[curr.project] = { project: curr.project, spendValue: 0 };
      }
      acc[curr.project].spendValue += curr.spendValue;
      return acc;
    }, {}));

    const fSpendOverTime = (() => {
      if (selectedProject === 'All Projects') {
        return spendOverTime;
      }
      const totalSpendAllProjects = costBreakdown.reduce((sum, item) => sum + parseFloat(item.spend.replace(/[^0-9.-]+/g, "")), 0);
      const projectSpend = pCostBreakdown.reduce((sum, item) => sum + item.spendValue, 0);
      if (totalSpendAllProjects === 0) return spendOverTime.map(d => ({ ...d, spend: 0 }));
      const projectRatio = projectSpend / totalSpendAllProjects;
      return spendOverTime.map(d => ({
        ...d,
        spend: Math.round(d.spend * projectRatio),
      }));
    })();

    return {
      filteredCostBreakdown: fCostBreakdown,
      filteredAnomalies: fAnomalies,
      dynamicSummaryMetrics: dSummaryMetrics,
      parsedCostBreakdown: pCostBreakdown,
      spendByProject: sByProject,
      filteredSpendOverTime: fSpendOverTime,
    };
  }, [selectedProject]);

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1 style={{ margin: 0 }}>Dashboard</h1>
        <div className="dashboard-filter">
          <label htmlFor="project-filter">Filter by Project:</label>
          <select
            id="project-filter"
            value={selectedProject}
            onChange={(e) => setSelectedProject(e.target.value)}
          >
            {projectNames.map(name => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        </div>
      </div>

      <section className="metrics-grid">
        {dynamicSummaryMetrics.map((metric) => (
          <MetricCard key={metric.title} title={metric.title} value={metric.value} change={metric.change} />
        ))}
      </section>

      <section className="dashboard-section">
        <h2>Spend Overview (Month-to-Date)</h2>
        <div className="chart-wrapper">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={filteredSpendOverTime} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
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
          rows={filteredCostBreakdown}
        />
      </section>

      <section className="dashboard-section">
        <h2>Recent Anomalies & Alerts</h2>
        <DataTable
          headers={['Timestamp', 'Service', 'Description', 'Severity']}
          dataKeys={['timestamp', 'service', 'description', 'severity']}
          rows={filteredAnomalies}
        />
      </section>
    </div>
  );
}

export default Dashboard;