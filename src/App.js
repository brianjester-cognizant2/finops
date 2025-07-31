import React from 'react';
import { Routes, Route } from 'react-router-dom';
import './App.css';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import CostAnalysis from './pages/CostAnalysis';
import Monitoring from './pages/Monitoring';
import Recommendations from './pages/Recommendations';

function App() {
  return (
    <div className="App">
      <Sidebar />
      <div className="main-content">
        <Header />
        <main>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/cost-analysis" element={<CostAnalysis />} />
            <Route path="/monitoring" element={<Monitoring />} />
            <Route path="/recommendations" element={<Recommendations />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

export default App;
