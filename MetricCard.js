import React from 'react';
import './MetricCard.css';

function MetricCard({ title, value, change, severity }) {
  const severityClass = severity ? `severity-${severity}` : '';

  return (
    <div className={`metric-card ${severityClass}`}>
      <h3 className="metric-title">{title}</h3>
      <p className="metric-value">{value}</p>
      {change && <p className="metric-change">{change}</p>}
    </div>
  );
}

export default MetricCard;