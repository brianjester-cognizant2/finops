import React from 'react';
import './Header.css';

function Header() {
  return (
    <header className="header">
      <div className="header-title">
        <h1>GenAI Cost Optimizer Dashboard</h1>
      </div>
      <div className="user-profile">
        <span>Welcome, Admin</span>
        <div className="avatar">A</div>
      </div>
    </header>
  );
}

export default Header;
