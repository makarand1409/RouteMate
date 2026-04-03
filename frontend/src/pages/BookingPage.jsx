import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import MapView from '../components/MapView';
import TripStatus from '../components/TripStatus';
import { runSimulation } from '../services/api';
import './BookingPage.css';

const MUMBAI_VEHICLES = [
  { id: 1, position: [19.0760, 72.8777], status: 'idle', name: 'Rahul K.' },
  { id: 2, position: [19.0820, 72.8850], status: 'idle', name: 'Amit S.' },
  { id: 3, position: [19.0700, 72.8700], status: 'idle', name: 'Priya M.' },
  { id: 4, position: [19.0780, 72.8900], status: 'idle', name: 'Vijay R.' },
  { id: 5, position: [19.0850, 72.8750], status: 'idle', name: 'Sneha T.' },
];

const POOLER_NAMES = ['Nisha P.', 'Karan D.', 'Meera A.', 'Arjun N.', 'Fatima R.'];

function BookingPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { logout } = useAuth();
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
  const [rideType, setRideType] = useState('x');
  const [poolCandidates, setPoolCandidates] = useState([]);

  // Poll /api/vehicles/live every 500ms to get updated vehicle positions
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch('http://localhost:8000/api/vehicles/live');
        if (response.ok) {
          const data = await response.json();
          // Convert backend vehicle format to frontend format
          const updatedVehicles = data.vehicles.map((v, idx) => ({
            id: v.vehicle_id,
            position: [v.location[0], v.location[1]],
            status: 'active',
            name: MUMBAI_VEHICLES[idx % MUMBAI_VEHICLES.length].name,
            occupancy: v.occupancy,
            queue_length: v.queue_length,
          }));
          if (updatedVehicles.length > 0) {
            setVehicles(updatedVehicles);
          }
        }
      } catch (error) {
        // Silently fail - backend might not be running yet
        // Keep using local vehicle data
      }
    }, 500);

    return () => clearInterval(interval);
  }, []);

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

  const handleBookRide = async () => {
    if (!pickup || !dropoff) {
      alert('Please select both pickup and dropoff on the map!');
      return;
    }

    // Run a quick 1-step simulation on the backend with the chosen policy
    let assigned;
    try {
      const data = await runSimulation({ policy, numSteps: 50 });
      // Pick the vehicle that served the most customers as "assigned"
      const best = data.vehicles.reduce((a, b) =>
        b.total_served > a.total_served ? b : a, data.vehicles[0]);
      assigned = {
        ...vehicles.find(v => v.id === best.vehicle_id) || vehicles[0],
        backendData: data,
      };
    } catch {
      // Fallback to client-side nearest if backend is unreachable
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
    }

    const livePoolers = vehicles
      .filter(v => v.id !== assigned?.id)
      .slice(0, 2)
      .map((v, idx) => ({
        name: POOLER_NAMES[idx] || `Rider ${idx + 1}`,
        pickupEtaMin: v.id + 2,
      }));

    const selectedPoolers = rideType === 'pool' ? livePoolers : [];
    setPoolCandidates(selectedPoolers);

    setAssignedVehicle({
      ...assigned,
      rideType,
      poolers: selectedPoolers,
    });
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
    setRideType('x');
    setPoolCandidates([]);
  };

  const getDistance = () => {
    if (!pickup || !dropoff) return null;
    return (Math.sqrt(
      Math.pow((pickup.lat - dropoff.lat) * 111, 2) +
      Math.pow((pickup.lng - dropoff.lng) * 111, 2)
    ) * 1000).toFixed(0);
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
          <button className="policy-info-btn" onClick={() => { logout(); navigate('/'); }}>
            Logout
          </button>
        </div>
      </nav>

      <div className="booking-layout">
        {/* Left Panel */}
        <div className="booking-left">

          {/* Policy Selector - Always Visible */}
          <div className="policy-panel-always">
            <h3>🧠 Choose Your Policy</h3>
            <p>How should we match your ride?</p>
            <div className="policy-options">
              <div
                className={`policy-option ${policy === 'greedy' ? 'active' : ''}`}
                onClick={() => setPolicy('greedy')}
              >
                <span>📍</span>
                <div>
                  <p>Greedy</p>
                  <small>Nearest vehicle</small>
                </div>
              </div>
              <div
                className={`policy-option ${policy === 'random' ? 'active' : ''}`}
                onClick={() => setPolicy('random')}
              >
                <span>🎲</span>
                <div>
                  <p>Random</p>
                  <small>Random match</small>
                </div>
              </div>
              <div
                className={`policy-option ${policy === 'ml' ? 'active' : ''}`}
                onClick={() => setPolicy('ml')}
              >
                <span>🤖</span>
                <div>
                  <p>ML / AI</p>
                  <small>Trained PPO 🏆</small>
                </div>
              </div>
            </div>
          </div>

          {/* Policy Selector - Hidden */}
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
                  <div
                    className={`ride-option ${rideType === 'x' ? 'selected' : ''}`}
                    onClick={() => setRideType('x')}
                  >
                    <span>🚗</span>
                    <div>
                      <p>RouteMATE X</p>
                      <small>4 seats • {getDistance()}m away</small>
                    </div>
                    <span className="fare">₹{(50 + (getDistance() / 1000) * 12).toFixed(0)}</span>
                  </div>
                  <div
                    className={`ride-option ${rideType === 'pool' ? 'selected' : ''}`}
                    onClick={() => setRideType('pool')}
                  >
                    <span>🚕</span>
                    <div>
                      <p>RouteMATE Pool</p>
                      <small>Share & save • ML matched</small>
                    </div>
                    <span className="fare">₹{Math.floor((50 + (getDistance() / 1000) * 12) * 0.6)}</span>
                  </div>

                  {rideType === 'pool' && (
                    <div className="pool-preview">
                      <p className="pool-preview-title">Likely poolers on this route</p>
                      {poolCandidates.length > 0 ? (
                        poolCandidates.map((rider, idx) => (
                          <div className="pooler-row" key={`${rider.name}-${idx}`}>
                            <span>👤 {rider.name}</span>
                            <small>pickup in ~{rider.pickupEtaMin} min</small>
                          </div>
                        ))
                      ) : (
                        <small>Poolers will be matched at booking time.</small>
                      )}
                    </div>
                  )}
                </div>
              )}

              <div className="action-buttons">
                <button
                  className="book-btn"
                  onClick={handleBookRide}
                  disabled={!pickup || !dropoff}
                >
                  Book {rideType === 'pool' ? '🚕 Pool' : '🚗 X'} Ride
                  {' '}
                  ({policy === 'ml' ? '🤖 ML' : policy === 'greedy' ? '📍 Greedy' : '🎲 Random'})
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
              rideType={rideType}
              poolers={assignedVehicle?.poolers || []}
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