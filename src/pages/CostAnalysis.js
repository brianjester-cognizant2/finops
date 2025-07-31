import React from 'react';
import PlaceholderPage from './PlaceholderPage';
import { placeholderPages } from '../data/mockData';

function CostAnalysis() {
  const { title, description } = placeholderPages.costAnalysis;
  return <PlaceholderPage title={title} description={description} />;
}

export default CostAnalysis;
