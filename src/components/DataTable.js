import React from 'react';
import './DataTable.css';

function DataTable({ headers, rows, dataKeys }) {
  return (
    <div className="data-table-container">
      <table className="data-table">
        <thead>
          <tr>
            {headers.map((header, index) => (
              <th key={index}>{header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {dataKeys.map((key, cellIndex) => (
                <td key={cellIndex}>{row[key]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default DataTable;
