import React from 'react';
import './TripStatus.css';

function TripStatus({ status, vehicle, pickup, dropoff, policy, onReset }) {
  const steps = [
    { key: 'assigned', label: 'Driver Assigned', icon: '✅' },
    { key: 'picking_up', label: 'Driver Coming', icon: '🚗' },
    { key: 'in_progress', label: 'Trip In Progress', icon: '🛣️' },
    { key: 'completed', label: 'Trip Completed', icon: '🎉' },
  ];

  const currentIndex = steps.findIndex(s => s.key === status);

  const getDistance = () => {
    if (!pickup || !dropoff) return '0';
    return (Math.sqrt(
      Math.pow((pickup.lat - dropoff.lat) * 111, 2) +
      Math.pow((pickup.lng - dropoff.lng) * 111, 2)
    ) * 1000).toFixed(0);
  };

  const getFare = () => (50 + (getDistance() / 1000) * 12).toFixed(0);

  const policyLabels = {
    random: '🎲 Random',
    greedy: '📍 Greedy',
    ml: '🤖 ML/AI'
  };

  return (
    <div className="trip-status">

      {/* Driver Card */}
      {vehicle && (
        <div className="driver-card">
          <div className="driver-avatar">
            {vehicle.id % 2 === 0 ? '👨' : '👩'}
          </div>
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

      {/* Policy Badge */}
      <div className="policy-badge">
        Assigned via {policyLabels[policy] || policy}
      </div>

      {/* Progress Steps */}
      <div className="steps">
        {steps.map((step, index) => (
          <div
            key={step.key}
            className={`step ${index < currentIndex ? 'done' :
              index === currentIndex ? 'current' : 'pending'}`}
          >
            <div className="step-icon">{step.icon}</div>
            <div className="step-label">{step.label}</div>
            {index === currentIndex && (
              <div className="step-pulse"></div>
            )}
          </div>
        ))}
      </div>

      {/* Status Message */}
      <div className="status-msg">
        {status === 'assigned' && `🎯 ${vehicle?.name} is getting ready...`}
        {status === 'picking_up' && `🚗 ${vehicle?.name} is on the way!`}
        {status === 'in_progress' && `🛣️ Heading to your destination...`}
        {status === 'completed' && `🎉 You have arrived! Great ride!`}
      </div>

      {/* Trip Summary (on complete) */}
      {status === 'completed' && (
        <div className="trip-summary">
          <h4>Trip Summary</h4>
          <div className="summary-row">
            <span>Distance</span>
            <span>{getDistance()}m</span>
          </div>
          <div className="summary-row">
            <span>Fare</span>
            <span>₹{getFare()}</span>
          </div>
          <div className="summary-row">
            <span>Policy</span>
            <span>{policyLabels[policy]}</span>
          </div>
          <div className="summary-row">
            <span>Driver</span>
            <span>{vehicle?.name}</span>
          </div>
          <button className="new-ride-btn" onClick={onReset}>
            + Book New Ride
          </button>
        </div>
      )}
    </div>
  );
}

export default TripStatus;