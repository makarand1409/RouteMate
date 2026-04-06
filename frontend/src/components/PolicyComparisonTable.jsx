import React from 'react';

function PolicyComparisonTable({ policyRows = [] }) {
  const rows = policyRows.length > 0
    ? policyRows
    : [
        { key: 'greedy', label: 'Greedy', vehicleId: '-', etaMin: 0, distanceKm: 0, isMl: false },
        { key: 'random', label: 'Random', vehicleId: '-', etaMin: 0, distanceKm: 0, isMl: false },
        { key: 'ml', label: 'ML', vehicleId: '-', etaMin: 0, distanceKm: 0, isMl: true },
      ];

  return (
    <div className="battle-grid">
      {rows.map((row) => (
        <div key={row.key} className={`battle-policy-card ${row.isMl ? 'ml-highlight' : ''}`}>
          <div className="battle-policy-header">
            <strong>{row.label}{row.isMl ? ' STAR' : ''}</strong>
          </div>
          <div className="battle-policy-line">Vehicle: V{row.vehicleId}</div>
          <div className="battle-policy-line">ETA: {Number(row.etaMin).toFixed(1)} min</div>
          <div className="battle-policy-line">Distance: {Number(row.distanceKm).toFixed(2)} km</div>
        </div>
      ))}
    </div>
  );
}

export default PolicyComparisonTable;
