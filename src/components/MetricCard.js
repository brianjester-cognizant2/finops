import React from 'react';
import './MetricCard.css';

function MetricCard({ title, value, change }) {
  const changeClass = change && (change.startsWith('+') ? 'positive' : (change.startsWith('-') ? 'negative' : ''));
  return (
    <div className="metric-card">
      <h3 className="metric-title">{title}</h3>
      <p className="metric-value">{value}</p>
      {change && <p className={`metric-change ${changeClass}`}>{change}</p>}
    </div>
  );
}

export default MetricCard;
