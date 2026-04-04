import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import MapView from '../components/MapView';
import TripStatus from '../components/TripStatus';
import api from '../services/api';
import './BookingPage.css';

function BookingPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const { pickup: initialPickup, dropoff: initialDropoff } = location.state || {};

  const [pickup, setPickup] = useState(null);
  const [dropoff, setDropoff] = useState(null);
  const [pickupText, setPickupText] = useState(initialPickup || '');
  const [dropoffText, setDropoffText] = useState(initialDropoff || '');
  const [selectingMode, setSelectingMode] = useState(null);
  const [vehicles, setVehicles] = useState([]);
  const [assignedVehicle, setAssignedVehicle] = useState(null);
  const [tripStatus, setTripStatus] = useState(null);
  const [policy, setPolicy] = useState('greedy');
  const [showPolicyInfo, setShowPolicyInfo] = useState(false);
  const [loading, setLoading] = useState(false);
  const [currentRideId, setCurrentRideId] = useState(null);
  const [routeGeometry, setRouteGeometry] = useState([]);
  const [sharedRiders, setSharedRiders] = useState([]);
  const [liveVehiclePosition, setLiveVehiclePosition] = useState(null);
  const [liveSavings, setLiveSavings] = useState(null);
  const [rideProgress, setRideProgress] = useState(0);
  const [comparisonData, setComparisonData] = useState(null);
  const [compareLoading, setCompareLoading] = useState(false);

  const geocodeHydratedRef = useRef({ pickup: false, dropoff: false });
  const currentRideRef = useRef(null);
  const authUserId = user?.uid || user?.id || user?.localId || null;

  const resolveSelfTripStatus = useCallback((riders = []) => {
    const self = riders.find((r) => r.user_id === authUserId);
    if (!self) return null;
    if (self.status === 'completed') return 'completed';
    if (self.status === 'onboard') return 'in_progress';
    if (self.status === 'awaiting_pickup') return 'picking_up';
    return 'assigned';
  }, [authUserId]);

  useEffect(() => {
    const hydrateFromAddress = async () => {
      try {
        if (pickupText && !pickup && !geocodeHydratedRef.current.pickup) {
          const pickupGeo = await api.geocodeAddress(pickupText);
          setPickup({ lat: pickupGeo.lat, lng: pickupGeo.lng });
          geocodeHydratedRef.current.pickup = true;
        }
        if (dropoffText && !dropoff && !geocodeHydratedRef.current.dropoff) {
          const dropoffGeo = await api.geocodeAddress(dropoffText);
          setDropoff({ lat: dropoffGeo.lat, lng: dropoffGeo.lng });
          geocodeHydratedRef.current.dropoff = true;
        }
      } catch (_error) {
        // Keep map pin selection as fallback.
      }
    };

    hydrateFromAddress();
  }, [pickupText, dropoffText, pickup, dropoff]);

  useEffect(() => {
    if (!authUserId) return undefined;

    const wsUrl = api.getRideWebSocketUrl(authUserId);
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        const riderIds = (payload?.riders || []).map((r) => r.user_id);
        const isMyRide = Boolean(
          (currentRideRef.current && payload?.ride_id === currentRideRef.current) ||
          riderIds.includes(authUserId)
        );

        if (!isMyRide) return;

        if (payload?.ride_id) {
          currentRideRef.current = payload.ride_id;
          setCurrentRideId(payload.ride_id);
        }

        if (Array.isArray(payload?.route?.geometry)) {
          setRouteGeometry(payload.route.geometry);
        }

        if (payload?.vehicle_location) {
          setLiveVehiclePosition(payload.vehicle_location);
        }

        if (Array.isArray(payload?.riders)) {
          setSharedRiders(payload.riders);
          const selfStatus = resolveSelfTripStatus(payload.riders);
          if (selfStatus) setTripStatus(selfStatus);
        }

        if (payload?.savings) {
          setLiveSavings(payload.savings);
        }

        if (typeof payload?.progress_percent === 'number') {
          setRideProgress(payload.progress_percent);
        }

        if (payload?.type === 'carpool_matched') {
          alert('Carpool matched! Shared ride confirmed.');
        }

        if (
          payload?.type === 'ride_event' &&
          payload?.event === 'dropoff_completed' &&
          payload?.user_id === authUserId
        ) {
          setRideProgress(100);
          setTripStatus('completed');
        }

        if (payload?.type === 'ride_completed') {
          setRideProgress(100);
          setTripStatus('completed');
        }
      } catch (_error) {
        // Ignore malformed payloads.
      }
    };

    const heartbeat = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send('ping');
      }
    }, 15000);

    return () => {
      clearInterval(heartbeat);
      ws.close();
    };
  }, [authUserId, resolveSelfTripStatus]);

  useEffect(() => {
    if (!currentRideId) return undefined;

    let cancelled = false;
    const interval = setInterval(async () => {
      try {
        const state = await api.getRideState(currentRideId);
        if (cancelled || !state?.ride) return;

        const ride = state.ride;
        if (ride?.vehicle_location) {
          setLiveVehiclePosition(ride.vehicle_location);
        }
        if (Array.isArray(ride?.route?.geometry)) {
          setRouteGeometry(ride.route.geometry);
        }
        if (Array.isArray(ride?.riders)) {
          setSharedRiders(ride.riders);
          const selfStatus = resolveSelfTripStatus(ride.riders);
          if (selfStatus) setTripStatus(selfStatus);
        }
        if (typeof ride?.progress_percent === 'number') {
          setRideProgress(ride.progress_percent);
        }
        if (ride?.savings) {
          setLiveSavings(ride.savings);
        }

        if (state.status === 'completed') {
          setRideProgress(100);
          setTripStatus('completed');
        }
      } catch (_error) {
        // polling is best-effort fallback
      }
    }, 2000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [currentRideId, resolveSelfTripStatus]);

  useEffect(() => {
    const fetchVehicles = async () => {
      try {
        const backendVehicles = await api.getVehicles();
        const mappedVehicles = backendVehicles.map((v) => {
          const latLng = api.gridToLatLng(v.location.x, v.location.y);
          return {
            id: v.vehicle_id,
            position: [latLng.lat, latLng.lng],
            status: v.is_idle ? 'idle' : 'busy',
            name: `Vehicle ${v.vehicle_id}`,
            occupancy: v.occupancy,
            capacity: v.capacity,
            available: v.is_idle,
          };
        });
        setVehicles(mappedVehicles);
      } catch (_error) {
        setVehicles([]);
      }
    };

    fetchVehicles();
    const interval = setInterval(fetchVehicles, 2500);
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

    setLoading(true);

    try {
      const assignment = await api.requestRide(pickup, dropoff, policy, {
        userId: authUserId,
        userName: user?.name || '',
        userEmail: user?.email || '',
      });

      const fromFleet = vehicles.find((v) => v.id === assignment.vehicle_id);
      const fallbackPos = assignment?.vehicle_location
        ? [assignment.vehicle_location.lat, assignment.vehicle_location.lng]
        : [pickup.lat, pickup.lng];

      setAssignedVehicle({
        ...(fromFleet || {
          id: assignment.vehicle_id,
          name: `Vehicle ${assignment.vehicle_id}`,
          status: 'busy',
          position: fallbackPos,
        }),
        position: fallbackPos,
        eta: assignment.estimated_time,
        distance: assignment.distance,
        policyUsed: assignment.policy_used,
        rideId: assignment.ride_id,
        carpoolMatched: assignment.carpool_matched,
      });

      setCurrentRideId(assignment.ride_id || null);
      currentRideRef.current = assignment.ride_id || null;
      setRouteGeometry(Array.isArray(assignment?.route?.geometry) ? assignment.route.geometry : []);
      setLiveVehiclePosition(assignment?.vehicle_location || null);
      setSharedRiders(
        (assignment?.matched_riders || []).map((id) => ({
          user_id: id,
          name: id,
          status: id === authUserId ? 'awaiting_pickup' : 'onboard',
          pickup,
          dropoff,
        }))
      );
      setLiveSavings(assignment?.savings || null);
      setComparisonData(null);
      setRideProgress(0);
      setTripStatus(assignment.carpool_matched ? 'picking_up' : 'assigned');

      if (assignment.carpool_matched) {
        alert('Carpool matched! Shared ride confirmed.');
      }
    } catch (error) {
      alert(error.message || 'Failed to book ride');
    } finally {
      setLoading(false);
    }
  };

  const handleCompareRide = async () => {
    if (!currentRideId || !authUserId) return;
    setCompareLoading(true);
    try {
      const [comparison, rideState] = await Promise.all([
        api.getRideComparison(currentRideId, authUserId),
        api.getRideState(currentRideId),
      ]);
      setComparisonData(comparison);

      const ride = rideState?.ride || {};
      const me = (ride.riders || []).find((r) => r.user_id === authUserId);
      navigate('/dashboard', {
        state: {
          routeContext: {
            rideId: currentRideId,
            routeGeometry: ride?.route?.geometry || routeGeometry || [],
            pickup: me?.pickup || pickup,
            dropoff: me?.dropoff || dropoff,
            vehicleLocation: ride?.vehicle_location || liveVehiclePosition,
            comparison,
          },
        },
      });
    } catch (error) {
      alert(error.message || 'Failed to compare ride policies');
    } finally {
      setCompareLoading(false);
    }
  };

  const handleReset = () => {
    setPickup(null);
    setDropoff(null);
    setPickupText('');
    setDropoffText('');
    setTripStatus(null);
    setAssignedVehicle(null);
    setCurrentRideId(null);
    currentRideRef.current = null;
    setRouteGeometry([]);
    setSharedRiders([]);
    setLiveVehiclePosition(null);
    setLiveSavings(null);
    setComparisonData(null);
    setCompareLoading(false);
    setRideProgress(0);
    setSelectingMode(null);
    geocodeHydratedRef.current = { pickup: false, dropoff: false };
  };

  const getDistance = () => {
    if (!pickup || !dropoff) return null;
    return (
      Math.sqrt(Math.pow((pickup.lat - dropoff.lat) * 111, 2) + Math.pow((pickup.lng - dropoff.lng) * 111, 2)) * 1000
    ).toFixed(0);
  };

  const getFare = () => {
    const dist = getDistance();
    if (!dist) return null;
    return (50 + (dist / 1000) * 12).toFixed(0);
  };

  return (
    <div className="booking-page">
      <nav className="booking-nav">
        <div className="nav-left">
          <button className="back-btn" onClick={() => navigate('/')}>←</button>
          <div className="logo">RouteMATE</div>
        </div>
        <div className="nav-right">
          <button className="policy-info-btn" onClick={() => setShowPolicyInfo(!showPolicyInfo)}>
            🧠 Policy: {policy.toUpperCase()}
          </button>
          <button
            className="policy-info-btn"
            onClick={async () => {
              await logout();
              navigate('/');
            }}
          >
            Logout
          </button>
        </div>
      </nav>

      <div className="booking-layout">
        <div className="booking-left">
          {showPolicyInfo && (
            <div className="policy-panel">
              <h3>🧠 Assignment Policy</h3>
              <p>Choose how vehicles get assigned:</p>
              <div className="policy-options">
                <div className={`policy-option ${policy === 'random' ? 'active' : ''}`} onClick={() => { setPolicy('random'); setShowPolicyInfo(false); }}>
                  <span>🎲</span>
                  <div><p>Random</p><small>Picks any vehicle randomly</small></div>
                </div>
                <div className={`policy-option ${policy === 'greedy' ? 'active' : ''}`} onClick={() => { setPolicy('greedy'); setShowPolicyInfo(false); }}>
                  <span>📍</span>
                  <div><p>Greedy</p><small>Always nearest vehicle</small></div>
                </div>
                <div className={`policy-option ${policy === 'ml' ? 'active' : ''}`} onClick={() => { setPolicy('ml'); setShowPolicyInfo(false); }}>
                  <span>🤖</span>
                  <div><p>ML / AI</p><small>Trained DQN model</small></div>
                </div>
              </div>
            </div>
          )}

          {!tripStatus ? (
            <div className="location-panel">
              <h2>Your ride</h2>
              <div className="input-group">
                <div className="input-row">
                  <span className="dot green-dot"></span>
                  <input type="text" placeholder="Pickup location" value={pickupText} onChange={(e) => setPickupText(e.target.value)} className="location-input" readOnly />
                  <button className={`pin-btn ${selectingMode === 'pickup' ? 'active' : ''}`} onClick={() => setSelectingMode(selectingMode === 'pickup' ? null : 'pickup')}>
                    {selectingMode === 'pickup' ? '✕' : '📍'}
                  </button>
                </div>
                <div className="input-divider"></div>
                <div className="input-row">
                  <span className="dot black-dot"></span>
                  <input type="text" placeholder="Dropoff location" value={dropoffText} onChange={(e) => setDropoffText(e.target.value)} className="location-input" readOnly />
                  <button className={`pin-btn ${selectingMode === 'dropoff' ? 'active' : ''}`} onClick={() => setSelectingMode(selectingMode === 'dropoff' ? null : 'dropoff')}>
                    {selectingMode === 'dropoff' ? '✕' : '🏁'}
                  </button>
                </div>
              </div>

              {selectingMode && <div className="map-hint">👆 Click on the map to set {selectingMode} location</div>}

              {vehicles.length > 0 && !pickup && !dropoff && (
                <div className="vehicle-status">🚗 {vehicles.filter((v) => v.status === 'idle').length} vehicles available nearby</div>
              )}

              {pickup && dropoff && (
                <div className="ride-options">
                  <div className="ride-option selected">
                    <span>🚗</span>
                    <div><p>RouteMATE X</p><small>4 seats • {getDistance()}m away</small></div>
                    <span className="fare">₹{getFare()}</span>
                  </div>
                </div>
              )}

              <div className="action-buttons">
                <button className="book-btn" onClick={handleBookRide} disabled={!pickup || !dropoff || loading}>
                  {loading ? '⏳ Booking...' : `Book ${policy === 'ml' ? '🤖 ML' : policy === 'greedy' ? '📍 Greedy' : '🎲 Random'} Ride`}
                </button>
                {(pickup || dropoff) && <button className="reset-btn" onClick={handleReset}>↺</button>}
              </div>
            </div>
          ) : (
            <TripStatus
              status={tripStatus}
              vehicle={assignedVehicle}
              pickup={pickup}
              dropoff={dropoff}
              policy={policy}
              savings={liveSavings}
              riders={sharedRiders}
              progress={rideProgress}
              rideId={currentRideId}
              comparisonData={comparisonData}
              compareLoading={compareLoading}
              onCompare={handleCompareRide}
              onReset={handleReset}
            />
          )}
        </div>

        <div className="booking-map">
          <MapView
            pickup={pickup}
            dropoff={dropoff}
            vehicles={vehicles}
            assignedVehicle={assignedVehicle}
            onMapClick={handleMapClick}
            selectingMode={selectingMode}
            center={[19.0760, 72.8777]}
            routeGeometry={routeGeometry}
            sharedRiders={sharedRiders}
            liveVehiclePosition={liveVehiclePosition}
          />
        </div>
      </div>
    </div>
  );
}

export default BookingPage;
