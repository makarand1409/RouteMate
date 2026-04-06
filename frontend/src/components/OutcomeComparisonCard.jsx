import React from 'react';

function OutcomeComparisonCard({ data }) {
  if (!data) return null;

  return (
    <div className="replay-outcome-card">
      <h4>Outcome Comparison</h4>

      <div className="replay-kpi-strip">
        <div className="replay-kpi-pill">Riders: {data.ml.ridersServed}</div>
        <div className="replay-kpi-pill">Distance: {data.ml.distanceKm.toFixed(2)} km</div>
        <div className="replay-kpi-pill">Occupancy: {data.ml.occupancyPct.toFixed(0)}%</div>
        <div className="replay-kpi-pill">Efficiency: +{data.efficiencyGainPct.toFixed(1)}%</div>
      </div>

      <div className="replay-outcome-grid">
        <div className="replay-outcome-col greedy">
          <h5>Greedy</h5>
          <p>Riders served: {data.greedy.ridersServed}</p>
          <p>Total distance: {data.greedy.distanceKm.toFixed(2)} km</p>
          <p>Occupancy: {data.greedy.occupancyPct.toFixed(0)}%</p>
        </div>
        <div className="replay-outcome-col ml">
          <h5>ML</h5>
          <p>Riders served: {data.ml.ridersServed}</p>
          <p>Total distance: {data.ml.distanceKm.toFixed(2)} km</p>
          <p>Occupancy: {data.ml.occupancyPct.toFixed(0)}%</p>
        </div>
      </div>

      <div className="replay-final-line">{data.finalLine}</div>
      <div className="replay-local-global-note">
        Greedy optimizes for immediate pickup (local optimum). ML optimizes for overall fleet efficiency (global optimum).
      </div>
    </div>
  );
}

export default OutcomeComparisonCard;
