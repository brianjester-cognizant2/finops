import React, { useMemo } from 'react';
import MetricCard from '../components/MetricCard';
import DataTable from '../components/DataTable';
import { optimizationRecommendations, modelEfficiencyRankings } from '../data/mockData';
import './Recommendations.css';

function Recommendations() {
  const recommendationCounts = useMemo(() => {
    return optimizationRecommendations.reduce((acc, rec) => {
      const severity = rec.severity.toLowerCase();
      acc[severity] = (acc[severity] || 0) + 1;
      return acc;
    }, {});
  }, []);

  const recommendationMetrics = [
    { title: 'High Priority Issues', value: recommendationCounts.high || 0, severity: 'high' },
    { title: 'Medium Priority Issues', value: recommendationCounts.medium || 0, severity: 'medium' },
    { title: 'Low Priority Issues', value: recommendationCounts.low || 0, severity: 'low' },
  ];

  return (
    <div className="recommendations-page">
      <h1>Optimization Recommendations</h1>

      <section className="metrics-grid">
        {recommendationMetrics.map((metric) => (
          <MetricCard
            key={metric.title}
            title={metric.title}
            value={metric.value}
            severity={metric.severity}
          />
        ))}
      </section>

      <section className="dashboard-section">
        <h2>Model Efficiency Rankings</h2>
        <p>Comparative analysis of different models based on cost and performance metrics.</p>
        <DataTable
          headers={['Model', 'Cost per 1M Tokens', 'Avg. Latency', 'Success Rate']}
          dataKeys={['model', 'costPerMillionTokens', 'avgLatency', 'successRate']}
          rows={modelEfficiencyRankings}
        />
      </section>

      <section className="dashboard-section">
        <h2>Detailed Recommendations</h2>
        <DataTable
          headers={['ID', 'Category', 'Severity', 'Description']}
          dataKeys={['id', 'category', 'severity', 'description']}
          rows={optimizationRecommendations}
        />
      </section>
    </div>
  );
}

export default Recommendations;