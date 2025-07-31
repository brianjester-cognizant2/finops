import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import App from './App';

test('renders dashboard page by default', () => {
  render(
    <MemoryRouter>
      <App />
    </MemoryRouter>
  );
  const headingElement = screen.getByRole('heading', { name: /GenAI Cost Optimizer Dashboard/i });
  expect(headingElement).toBeInTheDocument();
});
