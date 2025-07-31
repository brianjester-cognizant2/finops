import React from 'react';
import PlaceholderPage from './PlaceholderPage';
import { placeholderPages } from '../data/mockData';

function Monitoring() {
  const { title, description } = placeholderPages.monitoring;
  return <PlaceholderPage title={title} description={description} />;
}

export default Monitoring;
