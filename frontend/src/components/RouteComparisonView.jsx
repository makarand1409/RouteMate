import React from 'react';

function RouteComparisonView({ data }) {
  if (!data) return null;

  const maxDistance = Math.max(data.greedy.distanceKm, data.ml.distanceKm, 0.001);
  const greedyWidth = (data.greedy.distanceKm / maxDistance) * 100;
  const mlWidth = (data.ml.distanceKm / maxDistance) * 100;

  return (
    <div className="replay-route-grid">
      <div className="replay-route-card greedy">
        <h4>Greedy Route</h4>
        <p>Local optimum: nearest immediate pickup</p>
        <div className="replay-route-bar-wrap">
          <div className="replay-route-bar greedy" style={{ width: `${greedyWidth}%` }} />
        </div>
        <p>Distance: {data.greedy.distanceKm.toFixed(2)} km</p>
        <p>Riders served: {data.greedy.ridersServed}</p>
      </div>

      <div className="replay-route-card ml">
        <h4>ML Route</h4>
        <p>Global optimum: better fleet decision</p>
        <div className="replay-route-bar-wrap">
          <div className="replay-route-bar ml" style={{ width: `${mlWidth}%` }} />
        </div>
        <p>Distance: {data.ml.distanceKm.toFixed(2)} km</p>
        <p>Riders served: {data.ml.ridersServed}</p>
      </div>
    </div>
  );
}

export default RouteComparisonView;
