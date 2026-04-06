import React from 'react';
import './TripStatus.css';

function TripStatus({
  status,
  vehicle,
  pickup,
  dropoff,
  policy,
  onReset,
  savings,
  riders = [],
  progress = 0,
  rideId = null,
  comparisonData = null,
  compareLoading = false,
  onCompare,
  battleSnapshot = null,
}) {
  const steps = [
    { key: 'assigned', label: 'Driver Assigned', icon: '✅' },
    { key: 'picking_up', label: 'Driver Coming', icon: '🚗' },
    { key: 'in_progress', label: 'Trip In Progress', icon: '🛣️' },
    { key: 'completed', label: 'Trip Completed', icon: '🎉' },
  ];

  const currentIndex = steps.findIndex((s) => s.key === status);

  const getDistance = () => {
    if (!pickup || !dropoff) return '0';
    return (
      Math.sqrt(Math.pow((pickup.lat - dropoff.lat) * 111, 2) + Math.pow((pickup.lng - dropoff.lng) * 111, 2)) * 1000
    ).toFixed(0);
  };

  const getFare = () => (50 + (Number(getDistance()) / 1000) * 12).toFixed(0);

  const policyLabels = {
    random: '🎲 Random',
    greedy: '📍 Greedy',
    ml: '🤖 ML/AI',
  };

  const resolvedPolicyBadge = (() => {
    if (battleSnapshot?.fallback) return '🚗 Policy: Greedy (ML fallback)';
    if (String(policy || '').toLowerCase() === 'ml') return '🚗 Policy: ML (Optimized)';
    return `Assigned via ${policyLabels[policy] || policy}`;
  })();

  return (
    <div className="trip-status">
      {vehicle && (
        <div className="driver-card">
          <div className="driver-avatar">{vehicle.id % 2 === 0 ? '👨' : '👩'}</div>
          <div className="driver-info">
            <p className="driver-name">{vehicle.name}</p>
            <small>Vehicle #{vehicle.id} • ⭐ 4.{vehicle.id + 5}</small>
          </div>
          <div className="driver-eta">
            <p>{status === 'completed' ? '✅' : '⏱️'}</p>
            <small>{status === 'completed' ? 'Done' : `${vehicle.id + 1} min`}</small>
          </div>
        </div>
      )}

      <div className="policy-badge">{resolvedPolicyBadge}</div>

      {rideId && (
        <div className="policy-badge" style={{ marginTop: 8 }}>
          Ride ID: {rideId.slice(0, 8)} • Riders: {riders.length || 1} / 3
        </div>
      )}

      {(status === 'picking_up' || status === 'in_progress') && (
        <div className="policy-badge" style={{ marginTop: 8 }}>
          Live Progress: {progress.toFixed(0)}%
        </div>
      )}

      <div className="steps">
        {steps.map((step, index) => (
          <div
            key={step.key}
            className={`step ${index < currentIndex ? 'done' : index === currentIndex ? 'current' : 'pending'}`}
          >
            <div className="step-icon">{step.icon}</div>
            <div className="step-label">{step.label}</div>
            {index === currentIndex && <div className="step-pulse"></div>}
          </div>
        ))}
      </div>

      <div className="status-msg">
        {status === 'assigned' && `🎯 ${vehicle?.name} is getting ready...`}
        {status === 'picking_up' && `🚗 ${vehicle?.name} is on the way!`}
        {status === 'in_progress' && `🛣️ Heading to your destination...`}
        {status === 'completed' && `🎉 You have arrived! Great ride!`}
      </div>

      {status === 'completed' && (
        <div className="trip-summary">
          <h4>Trip Summary</h4>
          <div className="summary-row"><span>Distance</span><span>{getDistance()}m</span></div>
          <div className="summary-row"><span>Fare</span><span>₹{getFare()}</span></div>
          <div className="summary-row"><span>Policy</span><span>{policyLabels[policy]}</span></div>
          <div className="summary-row"><span>Driver</span><span>{vehicle?.name}</span></div>

          {savings && (
            <>
              <div className="summary-row"><span>Time Saved</span><span>{savings.time_saved_per_rider_min || 0} min</span></div>
              <div className="summary-row"><span>Money Saved</span><span>₹{savings.money_saved_per_rider || 0}</span></div>
              <div className="summary-row"><span>Pooling Discount</span><span>{savings.pooling_discount_percent || 0}%</span></div>
            </>
          )}

          <button className="compare-btn" onClick={onCompare} disabled={compareLoading}>
            {compareLoading ? 'Comparing...' : 'Compare Greedy vs RL (Dashboard)'}
          </button>

          {comparisonData && (
            <div className="comparison-card">
              <h4>Policy Comparison (This Ride)</h4>
              <div className="summary-row"><span>Greedy Cost</span><span>₹{comparisonData.greedy.estimated_cost}</span></div>
              <div className="summary-row"><span>RL Cost</span><span>₹{comparisonData.rl.estimated_cost}</span></div>
              <div className="summary-row"><span>Distance Gain</span><span>{comparisonData.improvement.distance_percent}%</span></div>
              <div className="summary-row"><span>Time Gain</span><span>{comparisonData.improvement.time_percent}%</span></div>
              <div className="summary-row"><span>Cost Gain</span><span>{comparisonData.improvement.cost_percent}%</span></div>
              <div className="status-msg" style={{ marginTop: 10 }}>🏆 {comparisonData.message}</div>
            </div>
          )}

          <button className="new-ride-btn" onClick={onReset}>+ Book New Ride</button>
        </div>
      )}
    </div>
  );
}

export default TripStatus;
