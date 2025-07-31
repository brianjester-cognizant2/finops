import React from 'react';
import PlaceholderPage from './PlaceholderPage';
import { placeholderPages } from '../data/mockData';

function Recommendations() {
  const { title, description } = placeholderPages.recommendations;
  return <PlaceholderPage title={title} description={description} />;
}

export default Recommendations;
