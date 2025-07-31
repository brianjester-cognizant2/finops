import React, { useState, useMemo } from 'react';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import MetricCard from '../components/MetricCard';
import DataTable from '../components/DataTable';
import { summaryMetrics, costBreakdown, recentAnomalies, spendOverTimeData } from '../data/mockData';
import './Dashboard.css';

const COLORS = ['#a855f7', '#34d399', '#f97316', '#8b5cf6', '#84cc16', '#facc15'];

function Dashboard() {
  const [selectedProject, setSelectedProject] = useState('All Projects');
  const [timeRange, setTimeRange] = useState('mtd');
  const [selectedTeam, setSelectedTeam] = useState('All Teams');
  const [selectedCloud, setSelectedCloud] = useState('All Clouds');

  const projectNames = useMemo(() => ['All Projects', ...new Set(costBreakdown.map(item => item.project))], []);
  const teamNames = useMemo(() => ['All Teams', ...new Set(costBreakdown.map(item => item.team))], []);
  const cloudNames = useMemo(() => ['All Clouds', ...new Set(costBreakdown.map(item => item.cloudProvider))], []);

  const {
    filteredCostBreakdown,
    filteredAnomalies,
    dynamicSummaryMetrics,
    parsedCostBreakdown,
    spendByProject,
    filteredSpendOverTime
  } = useMemo(() => {
    const fCostBreakdown = costBreakdown
      .filter(item => selectedProject === 'All Projects' || item.project === selectedProject)
      .filter(item => selectedTeam === 'All Teams' || item.team === selectedTeam)
      .filter(item => selectedCloud === 'All Clouds' || item.cloudProvider === selectedCloud);

    const allowedServices = new Set(fCostBreakdown.map(item => item.service));
    const fAnomalies = recentAnomalies.filter(anomaly => allowedServices.has(anomaly.service));

    const pCostBreakdown = fCostBreakdown.map(item => ({
      ...item,
      spendValue: parseFloat(item.spend.replace(/[^0-9.-]+/g, "")),
    }));

    const totalSpend = pCostBreakdown.reduce((sum, item) => sum + item.spendValue, 0);
    const dSummaryMetrics = [
      { title: 'Total Spend (Month-to-Date)', value: new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(totalSpend), change: summaryMetrics[0].change },
      { title: 'Forecasted Spend', value: new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(totalSpend * 1.75), change: summaryMetrics[1].change },
      { title: 'Active Projects', value: new Set(fCostBreakdown.map(p => p.project)).size },
      { ...summaryMetrics[3], value: Math.ceil(fCostBreakdown.length / 2) },
    ];

    const sByProject = Object.values(pCostBreakdown.reduce((acc, curr) => {
      if (!acc[curr.project]) {
        acc[curr.project] = { project: curr.project, spendValue: 0 };
      }
      acc[curr.project].spendValue += curr.spendValue;
      return acc;
    }, {}));

    const fSpendOverTime = (() => {
      const sliceData = (data) => {
        if (timeRange === '7d') {
          return data.slice(-7);
        }
        if (timeRange === '30d') {
          return data.slice(-30);
        }
        return data; // 'mtd'
      };

      if (selectedProject !== 'All Projects') {
        // Data for single project view is in format: [{ date, spend }]
        return sliceData(spendOverTimeData[selectedProject] || []);
      }

      // Data for "All Projects" view needs to be combined.
      // Format: [{ date, 'Project Alpha': 123, 'Project Beta': 456, ... }]
      const projectKeys = projectNames.filter(p => p !== 'All Projects');

      // Use the first project's data as the base for dates.
      const combinedData = spendOverTimeData[projectKeys[0]].map((dataPoint, index) => {
        const entry = { date: dataPoint.date };
        projectKeys.forEach(pKey => {
          entry[pKey] = spendOverTimeData[pKey][index]?.spend || 0;
        });
        return entry;
      });
      return sliceData(combinedData);
    })();

    return {
      filteredCostBreakdown: fCostBreakdown,
      filteredAnomalies: fAnomalies,
      dynamicSummaryMetrics: dSummaryMetrics,
      parsedCostBreakdown: pCostBreakdown,
      spendByProject: sByProject,
      filteredSpendOverTime: fSpendOverTime,
    };
  }, [selectedProject, timeRange, selectedTeam, selectedCloud, projectNames]);

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1 style={{ margin: 0 }}>Dashboard</h1>
        <div className="filters-container">
          <div className="dashboard-filter">
            <label htmlFor="timerange-filter">Time Range:</label>
            <select id="timerange-filter" value={timeRange} onChange={(e) => setTimeRange(e.target.value)}>
              <option value="mtd">Full Month</option>
              <option value="30d">Last 30 Days</option>
              <option value="7d">Last 7 Days</option>
            </select>
          </div>
          <div className="dashboard-filter">
            <label htmlFor="project-filter">Project:</label>
            <select id="project-filter" value={selectedProject} onChange={(e) => setSelectedProject(e.target.value)}>
              {projectNames.map(name => (<option key={name} value={name}>{name}</option>))}
            </select>
          </div>
          <div className="dashboard-filter">
            <label htmlFor="team-filter">Team:</label>
            <select id="team-filter" value={selectedTeam} onChange={(e) => setSelectedTeam(e.target.value)}>
              {teamNames.map(name => (<option key={name} value={name}>{name}</option>))}
            </select>
          </div>
          <div className="dashboard-filter">
            <label htmlFor="cloud-filter">Cloud:</label>
            <select id="cloud-filter" value={selectedCloud} onChange={(e) => setSelectedCloud(e.target.value)}>
              {cloudNames.map(name => (<option key={name} value={name}>{name}</option>))}
            </select>
          </div>
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
              {selectedProject === 'All Projects' ? (
                projectNames.filter(p => p !== 'All Projects').map((projectName, index) => (
                  <Line
                    key={projectName}
                    type="monotone"
                    dataKey={projectName}
                    stroke={COLORS[index % COLORS.length]}
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 6 }}
                  />
                ))
              ) : (
                <Line
                  type="monotone"
                  dataKey="spend"
                  name={selectedProject}
                  stroke={COLORS[0]}
                  strokeWidth={2}
                  activeDot={{ r: 8 }}
                />
              )}
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