import React from 'react';
import './PlaceholderPage.css';

function PlaceholderPage({ title, description }) {
  return (
    <div className="placeholder-page">
      <h2>{title}</h2>
      <p>{description}</p>
      <ul>
        <li>Detailed charts and graphs for cost breakdown.</li>
        <li>Filters for time range, project, team, and cloud provider.</li>
        <li>Data export functionality.</li>
      </ul>
    </div>
  );
}

export default PlaceholderPage;
