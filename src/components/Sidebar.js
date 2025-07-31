import React from 'react';
import { NavLink } from 'react-router-dom';
import './Sidebar.css';

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2>GenAI Opt</h2>
      </div>
      <nav className="sidebar-nav">
        <ul>
          <li>
            <NavLink to="/" className={({ isActive }) => (isActive ? 'active' : '')}>
              Dashboard
            </NavLink>
          </li>
          <li>
            <NavLink to="/cost-analysis" className={({ isActive }) => (isActive ? 'active' : '')}>
              Cost Analysis
            </NavLink>
          </li>
          <li>
            <NavLink to="/monitoring" className={({ isActive }) => (isActive ? 'active' : '')}>
              Monitoring & Alerts
            </NavLink>
          </li>
          <li>
            <NavLink to="/recommendations" className={({ isActive }) => (isActive ? 'active' : '')}>
              Recommendations
            </NavLink>
          </li>
        </ul>
      </nav>
    </aside>
  );
}

export default Sidebar;
