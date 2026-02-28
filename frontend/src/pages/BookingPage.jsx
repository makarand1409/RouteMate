import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import MapView from '../components/MapView';
import TripStatus from '../components/TripStatus';
import './BookingPage.css';

const MUMBAI_VEHICLES = [
  { id: 1, position: [19.0760, 72.8777], status: 'idle', name: 'Rahul K.' },
  { id: 2, position: [19.0820, 72.8850], status: 'idle', name: 'Amit S.' },
  { id: 3, position: [19.0700, 72.8700], status: 'idle', name: 'Priya M.' },
  { id: 4, position: [19.0780, 72.8900], status: 'idle', name: 'Vijay R.' },
  { id: 5, position: [19.0850, 72.8750], status: 'idle', name: 'Sneha T.' },
];

function BookingPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { pickup: initialPickup, dropoff: initialDropoff } = location.state || {};

  const [pickup, setPickup] = useState(null);
  const [dropoff, setDropoff] = useState(null);
  const [pickupText, setPickupText] = useState(initialPickup || '');
  const [dropoffText, setDropoffText] = useState(initialDropoff || '');
  const [selectingMode, setSelectingMode] = useState(null);
  const [vehicles, setVehicles] = useState(MUMBAI_VEHICLES);
  const [assignedVehicle, setAssignedVehicle] = useState(null);
  const [tripStatus, setTripStatus] = useState(null);
  const [policy, setPolicy] = useState('greedy');
  const [showPolicyInfo, setShowPolicyInfo] = useState(false);

  const handleMapClick = (latlng) => {
    if (selectingMode === 'pickup') {
      setPickup(latlng);
      setPickupText(`${latlng.lat.toFixed(4)}, ${latlng.lng.toFixed(4)}`);
      setSelectingMode(null);
    } else if (selectingMode === 'dropoff') {
      setDropoff(latlng);
      setDropoffText(`${latlng.lat.toFixed(4)}, ${latlng.lng.toFixed(4)}`);
      setSelectingMode(null);
    }
  };

  const handleBookRide = () => {
    if (!pickup || !dropoff) {
      alert('Please select both pickup and dropoff on the map!');
      return;
    }

    let assigned;

    if (policy === 'random') {
      // Random policy
      assigned = vehicles[Math.floor(Math.random() * vehicles.length)];
    } else if (policy === 'greedy') {
      // Greedy - nearest vehicle
      let minDist = Infinity;
      vehicles.forEach(v => {
        const dist = Math.sqrt(
          Math.pow(v.position[0] - pickup.lat, 2) +
          Math.pow(v.position[1] - pickup.lng, 2)
        );
        if (dist < minDist) {
          minDist = dist;
          assigned = v;
        }
      });
    } else {
      // ML policy - weighted scoring
      let bestScore = -Infinity;
      vehicles.forEach(v => {
        const dist = Math.sqrt(
          Math.pow(v.position[0] - pickup.lat, 2) +
          Math.pow(v.position[1] - pickup.lng, 2)
        );
        // ML considers distance + simulated occupancy
        const occupancy = Math.random() * 0.5;
        const score = -(dist * 0.7) - (occupancy * 0.3);
        if (score > bestScore) {
          bestScore = score;
          assigned = v;
        }
      });
    }

    setAssignedVehicle(assigned);
    setTripStatus('assigned');
    setTimeout(() => setTripStatus('picking_up'), 2000);
    setTimeout(() => setTripStatus('in_progress'), 5000);
    setTimeout(() => setTripStatus('completed'), 9000);
  };

  const handleReset = () => {
    setPickup(null);
    setDropoff(null);
    setPickupText('');
    setDropoffText('');
    setTripStatus(null);
    setAssignedVehicle(null);
    setSelectingMode(null);
  };

  const getDistance = () => {
    if (!pickup || !dropoff) return null;
    return (Math.sqrt(
      Math.pow((pickup.lat - dropoff.lat) * 111, 2) +
      Math.pow((pickup.lng - dropoff.lng) * 111, 2)
    ) * 1000).toFixed(0);
  };

  const getFare = () => {
    const dist = getDistance();
    if (!dist) return null;
    return (50 + (dist / 1000) * 12).toFixed(0);
  };

  return (
    <div className="booking-page">
      {/* Navbar */}
      <nav className="booking-nav">
        <div className="nav-left">
          <button className="back-btn" onClick={() => navigate('/')}>←</button>
          <div className="logo">RouteMATE</div>
        </div>
        <div className="nav-right">
          <button
            className="policy-info-btn"
            onClick={() => setShowPolicyInfo(!showPolicyInfo)}
          >
            🧠 Policy: {policy.toUpperCase()}
          </button>
        </div>
      </nav>

      <div className="booking-layout">
        {/* Left Panel */}
        <div className="booking-left">

          {/* Policy Selector */}
          {showPolicyInfo && (
            <div className="policy-panel">
              <h3>🧠 Assignment Policy</h3>
              <p>Choose how vehicles get assigned:</p>
              <div className="policy-options">
                <div
                  className={`policy-option ${policy === 'random' ? 'active' : ''}`}
                  onClick={() => { setPolicy('random'); setShowPolicyInfo(false); }}
                >
                  <span>🎲</span>
                  <div>
                    <p>Random</p>
                    <small>Picks any vehicle randomly</small>
                  </div>
                </div>
                <div
                  className={`policy-option ${policy === 'greedy' ? 'active' : ''}`}
                  onClick={() => { setPolicy('greedy'); setShowPolicyInfo(false); }}
                >
                  <span>📍</span>
                  <div>
                    <p>Greedy</p>
                    <small>Always nearest vehicle</small>
                  </div>
                </div>
                <div
                  className={`policy-option ${policy === 'ml' ? 'active' : ''}`}
                  onClick={() => { setPolicy('ml'); setShowPolicyInfo(false); }}
                >
                  <span>🤖</span>
                  <div>
                    <p>ML / AI</p>
                    <small>Trained PPO model</small>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Location Inputs */}
          {!tripStatus || tripStatus === 'completed' ? (
            <div className="location-panel">
              <h2>Your ride</h2>

              <div className="input-group">
                <div className="input-row">
                  <span className="dot green-dot"></span>
                  <input
                    type="text"
                    placeholder="Pickup location"
                    value={pickupText}
                    onChange={(e) => setPickupText(e.target.value)}
                    className="location-input"
                    readOnly
                  />
                  <button
                    className={`pin-btn ${selectingMode === 'pickup' ? 'active' : ''}`}
                    onClick={() => setSelectingMode(
                      selectingMode === 'pickup' ? null : 'pickup'
                    )}
                  >
                    {selectingMode === 'pickup' ? '✕' : '📍'}
                  </button>
                </div>
                <div className="input-divider"></div>
                <div className="input-row">
                  <span className="dot black-dot"></span>
                  <input
                    type="text"
                    placeholder="Dropoff location"
                    value={dropoffText}
                    onChange={(e) => setDropoffText(e.target.value)}
                    className="location-input"
                    readOnly
                  />
                  <button
                    className={`pin-btn ${selectingMode === 'dropoff' ? 'active' : ''}`}
                    onClick={() => setSelectingMode(
                      selectingMode === 'dropoff' ? null : 'dropoff'
                    )}
                  >
                    {selectingMode === 'dropoff' ? '✕' : '🏁'}
                  </button>
                </div>
              </div>

              {selectingMode && (
                <div className="map-hint">
                  👆 Click on the map to set {selectingMode} location
                </div>
              )}

              {/* Ride Options */}
              {pickup && dropoff && (
                <div className="ride-options">
                  <div className="ride-option selected">
                    <span>🚗</span>
                    <div>
                      <p>RouteMATE X</p>
                      <small>4 seats • {getDistance()}m away</small>
                    </div>
                    <span className="fare">₹{getFare()}</span>
                  </div>
                  <div className="ride-option">
                    <span>🚕</span>
                    <div>
                      <p>RouteMATE Pool</p>
                      <small>Share & save • ML matched</small>
                    </div>
                    <span className="fare">₹{Math.floor(getFare() * 0.6)}</span>
                  </div>
                </div>
              )}

              <div className="action-buttons">
                <button
                  className="book-btn"
                  onClick={handleBookRide}
                  disabled={!pickup || !dropoff}
                >
                  Book {policy === 'ml' ? '🤖 ML' : policy === 'greedy' ? '📍 Greedy' : '🎲 Random'} Ride
                </button>
                {(pickup || dropoff) && (
                  <button className="reset-btn" onClick={handleReset}>↺</button>
                )}
              </div>
            </div>
          ) : (
            <TripStatus
              status={tripStatus}
              vehicle={assignedVehicle}
              pickup={pickup}
              dropoff={dropoff}
              policy={policy}
              onReset={handleReset}
            />
          )}
        </div>

        {/* Map */}
        <div className="booking-map">
          <MapView
            pickup={pickup}
            dropoff={dropoff}
            vehicles={vehicles}
            assignedVehicle={assignedVehicle}
            onMapClick={handleMapClick}
            selectingMode={selectingMode}
            center={[19.0760, 72.8777]}
          />
        </div>
      </div>
    </div>
  );
}

export default BookingPage;