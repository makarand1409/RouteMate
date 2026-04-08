import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import MapView from '../components/MapView';
import TripStatus from '../components/TripStatus';
import PolicyBattleModal from '../components/PolicyBattleModal';
import LocationSuggestions from '../components/LocationSuggestions';
import PoolingSuggestion from '../components/PoolingSuggestion';
import CarTypeSelection from '../components/CarTypeSelection';
import CancelRideModal from '../components/CancelRideModal';
import AIDecisionBadge from '../components/AIDecisionBadge';
import DailyCommuteSuggestion from '../components/DailyCommuteSuggestion';
import api from '../services/api';
import './BookingPage.css';

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function distanceKm(a, b) {
  if (!a || !b) return 0;
  const latKm = (a.lat - b.lat) * 111;
  const lngKm = (a.lng - b.lng) * 111;
  return Math.sqrt(latKm * latKm + lngKm * lngKm);
}

function buildVehicleCandidates(vehicles, pickupPoint, carType = 'sedan') {
  if (!Array.isArray(vehicles) || vehicles.length === 0) {
    return [
      { id: 1, distanceKm: 0.8, etaMin: 2.8, nearbyRequests: 1, availableSeats: 2 },
      { id: 2, distanceKm: 1.2, etaMin: 3.8, nearbyRequests: 3, availableSeats: 3 },
      { id: 3, distanceKm: 0.5, etaMin: 2.0, nearbyRequests: 0, availableSeats: 1 },
    ];
  }

  const requiredSeats = carType === 'mini' ? 3 : carType === 'suv' ? 6 : 4;
  const eligibleVehicles = vehicles.filter((v) => Number(v.capacity || 4) >= requiredSeats);
  const source = eligibleVehicles.length > 0 ? eligibleVehicles : vehicles;

  return source.map((v) => {
    const vehiclePoint = { lat: v.position[0], lng: v.position[1] };
    const dKm = distanceKm(vehiclePoint, pickupPoint);
    const nearbyRequests = Math.abs(Math.round((pickupPoint.lat * 1000) + (pickupPoint.lng * 1000) + v.id * 7)) % 4;
    const availableSeats = Math.max(0, Number(v.capacity || 4) - Number(v.occupancy || 0));
    const etaMin = (dKm / 25) * 60 + 1;

    return {
      id: v.id,
      distanceKm: Number(dKm.toFixed(3)),
      etaMin: Number(etaMin.toFixed(2)),
      nearbyRequests,
      availableSeats,
    };
  });
}

function buildBattleSnapshot(vehicles, pickupPoint, carType = 'sedan') {
  const candidates = buildVehicleCandidates(vehicles, pickupPoint, carType);
  const greedy = candidates.reduce((best, c) => (!best || c.distanceKm < best.distanceKm ? c : best), null);

  const rand = seededRandom(42);
  const random = candidates[Math.floor(rand() * candidates.length)] || candidates[0];

  const mlScored = candidates.map((c) => ({
    ...c,
    score: -c.distanceKm + (c.nearbyRequests * 2) + c.availableSeats,
  }));
  const ml = mlScored.reduce((best, c) => (!best || c.score > best.score ? c : best), null);

  const scoreDiff = Number((ml.score - ((-greedy.distanceKm) + (greedy.nearbyRequests * 2) + greedy.availableSeats)).toFixed(3));
  const rawConfidence = sigmoid(scoreDiff) * 100;
  const confidence = Math.round(clamp(rawConfidence, 55, 95));

  const fallback = ml.etaMin > (greedy.etaMin + 1.5);
  const finalPolicy = fallback ? 'greedy' : 'ml';

  const pickupDelta = Number((ml.etaMin - greedy.etaMin).toFixed(1));
  const poolingProbability = Math.round(clamp(35 + (ml.nearbyRequests * 15) + (ml.availableSeats * 4), 15, 96));
  const expectedOccupancyGain = Number((0.2 + (ml.nearbyRequests * 0.35) + (ml.availableSeats * 0.08)).toFixed(1));

  const mlAgreesWithGreedy = ml.id === greedy.id;
  const positives = mlAgreesWithGreedy
    ? ['ML agrees with greedy - optimal decision', 'Improves trust with consistent selection', 'Balances local and fleet objectives']
    : ['Enables pooling with nearby riders', 'Improves system efficiency', 'Better seat utilization'];

  const tradeoff = pickupDelta > 0
    ? `Slightly higher pickup time (+${pickupDelta.toFixed(1)} min)`
    : 'No pickup-time tradeoff';

  const winnerLine = fallback
    ? 'Trophy: Greedy wins due to better immediate pickup time.'
    : `Trophy: ML wins this request because it improves overall system efficiency despite +${Math.max(0, pickupDelta).toFixed(1)} min pickup time.`;

  return {
    greedy,
    random,
    ml,
    confidence,
    fallback,
    winner: fallback ? 'greedy' : 'ml',
    finalPolicy,
    poolingProbability,
    expectedOccupancyGain,
    mlAgreesWithGreedy,
    winnerLine,
    reasoning: {
      positives,
      tradeoff,
    },
    policyRows: [
      { key: 'greedy', label: 'Greedy', vehicleId: greedy.id, etaMin: greedy.etaMin, distanceKm: greedy.distanceKm, isMl: false },
      { key: 'random', label: 'Random', vehicleId: random.id, etaMin: random.etaMin, distanceKm: random.distanceKm, isMl: false },
      { key: 'ml', label: 'ML', vehicleId: ml.id, etaMin: ml.etaMin, distanceKm: ml.distanceKm, isMl: true },
    ],
  };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

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
  const [battleOpen, setBattleOpen] = useState(false);
  const [battleLoading, setBattleLoading] = useState(false);
  const [battleSnapshot, setBattleSnapshot] = useState(null);
  const [battleCountdown, setBattleCountdown] = useState(0);
  const [battleAutoAssignAt, setBattleAutoAssignAt] = useState(null);

  // New feature states
  const [carType, setCarType] = useState('sedan');
  const [showPickupSuggestions, setShowPickupSuggestions] = useState(false);
  const [showDropoffSuggestions, setShowDropoffSuggestions] = useState(false);
  const [showPoolingSuggestion, setShowPoolingSuggestion] = useState(false);
  const [showCancelModal, setShowCancelModal] = useState(false);
  const [mapCenter, setMapCenter] = useState({ lat: 19.0760, lng: 72.8777 });
  const [mapZoom, setMapZoom] = useState(13);
  const [mapCursorLocation, setMapCursorLocation] = useState(null);
  const [lastMapClickLocation, setLastMapClickLocation] = useState(null);
  const [dailyCommuteSuggestion, setDailyCommuteSuggestion] = useState(null);

  const geocodeHydratedRef = useRef({ pickup: false, dropoff: false });
  const currentRideRef = useRef(null);
  const battleResolveRef = useRef(null);
  const battleTimerRef = useRef(null);
  const battleChosenRef = useRef(false);
  const cursorUpdateRef = useRef(null);
  const lastCursorPositionRef = useRef(null);
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

  useEffect(() => {
    if (!battleOpen || battleLoading || !battleAutoAssignAt) return undefined;

    const update = () => {
      const remaining = Math.max(0, Math.ceil((battleAutoAssignAt - Date.now()) / 1000));
      setBattleCountdown(remaining);
    };

    update();
    const intervalId = setInterval(update, 250);
    return () => clearInterval(intervalId);
  }, [battleOpen, battleLoading, battleAutoAssignAt]);

  useEffect(() => () => {
    if (battleTimerRef.current) {
      clearTimeout(battleTimerRef.current);
      battleTimerRef.current = null;
    }
    if (cursorUpdateRef.current) {
      clearTimeout(cursorUpdateRef.current);
      cursorUpdateRef.current = null;
    }
  }, []);

  const resolveBattleChoice = useCallback((policyChoice) => {
    if (battleChosenRef.current) return;
    battleChosenRef.current = true;

    if (battleTimerRef.current) {
      clearTimeout(battleTimerRef.current);
      battleTimerRef.current = null;
    }

    const resolve = battleResolveRef.current;
    battleResolveRef.current = null;
    setBattleAutoAssignAt(null);
    setBattleCountdown(0);

    if (resolve) {
      resolve(policyChoice);
    }
  }, []);

  const handleMapClick = async (latlng) => {
    const fallbackLabel = `${latlng.lat.toFixed(4)}, ${latlng.lng.toFixed(4)}`;
    let address = fallbackLabel;
    try {
      const response = await api.reverseGeocode(latlng.lat, latlng.lng, mapZoom);
      address = response.address || fallbackLabel;
    } catch (_error) {
      address = fallbackLabel;
    }

    const location = {
      lat: latlng.lat,
      lng: latlng.lng,
      address,
    };

    if (selectingMode === 'pickup') {
      setPickup(location);
      setPickupText(address);
      setSelectingMode(null);
      setLastMapClickLocation(location);
    } else if (selectingMode === 'dropoff') {
      setDropoff(location);
      setDropoffText(address);
      setSelectingMode(null);
      setLastMapClickLocation(location);
    } else {
      setLastMapClickLocation(location);
    }
  };

  const handleMapMove = (center) => {
    setMapCenter(center);
  };

  const handleCursorMove = (cursor) => {
    const last = lastCursorPositionRef.current;
    if (last && Math.abs(cursor.lat - last.lat) < 0.00015 && Math.abs(cursor.lng - last.lng) < 0.00015) {
      return;
    }
    lastCursorPositionRef.current = cursor;

    if (cursorUpdateRef.current) {
      clearTimeout(cursorUpdateRef.current);
    }

    cursorUpdateRef.current = setTimeout(() => {
      setMapCursorLocation(cursor);
      cursorUpdateRef.current = null;
    }, 180);
  };

  const assignRide = async (assignmentPolicy, snapshot = null) => {
    const assignment = await api.requestRide(pickup, dropoff, assignmentPolicy, {
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

    if (snapshot && assignment?.ride_id) {
      const persisted = { ...snapshot, rideId: assignment.ride_id, assignedPolicy: assignmentPolicy, savedAt: Date.now() };
      setBattleSnapshot(persisted);
      localStorage.setItem(`battleSnapshot:${assignment.ride_id}`, JSON.stringify(persisted));
    }

    if (assignment.carpool_matched) {
      alert('Carpool matched! Shared ride confirmed.');
    }
  };

  const handleBookRide = async () => {
    if (!pickup || !dropoff) {
      alert('Please select both pickup and dropoff on the map!');
      return;
    }

    setLoading(true);

    try {
      const snapshot = buildBattleSnapshot(vehicles, pickup, carType);
      setBattleSnapshot(snapshot);
      setBattleOpen(true);
      setBattleLoading(true);
      battleChosenRef.current = false;

      await sleep(1200);
      setBattleLoading(false);

      const autoMs = 7000;
      setBattleAutoAssignAt(Date.now() + autoMs);
      const selectedPolicy = await new Promise((resolve) => {
        battleResolveRef.current = resolve;
        battleTimerRef.current = setTimeout(() => resolveBattleChoice(snapshot.finalPolicy), autoMs);
      });

      setBattleOpen(false);
      setPolicy(selectedPolicy);
      await assignRide(selectedPolicy, snapshot);
    } catch (error) {
      alert(error.message || 'Failed to book ride');
    } finally {
      setBattleLoading(false);
      setBattleAutoAssignAt(null);
      setBattleCountdown(0);
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

      const storedSnapshotRaw = localStorage.getItem(`battleSnapshot:${currentRideId}`);
      let storedSnapshot = null;
      if (storedSnapshotRaw) {
        try {
          storedSnapshot = JSON.parse(storedSnapshotRaw);
        } catch (_error) {
          storedSnapshot = null;
        }
      }

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
            battleSnapshot: storedSnapshot || battleSnapshot,
            ridersServed: (ride?.riders || []).length,
            systemEfficiencyNote: (storedSnapshot || battleSnapshot)?.fallback
              ? 'Greedy was used for immediate pickup efficiency due to ML guardrail.'
              : 'ML selected a system-efficient dispatch balancing pooling potential and seat utilization.',
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
    setBattleOpen(false);
    setBattleLoading(false);
    setBattleSnapshot(null);
    setBattleCountdown(0);
    setBattleAutoAssignAt(null);
    setShowPickupSuggestions(false);
    setShowDropoffSuggestions(false);
    setShowPoolingSuggestion(false);
    setShowCancelModal(false);
    setCarType('sedan');
    battleChosenRef.current = true;
    battleResolveRef.current = null;
    if (battleTimerRef.current) {
      clearTimeout(battleTimerRef.current);
      battleTimerRef.current = null;
    }
    geocodeHydratedRef.current = { pickup: false, dropoff: false };
  };

  const handleMapZoom = (zoom) => {
    setMapZoom(zoom);
  };

  useEffect(() => {
    if (!authUserId) {
      setDailyCommuteSuggestion(null);
      return undefined;
    }

    let cancelled = false;
    const loadSuggestion = async () => {
      try {
        const data = await api.getUserRouteSuggestion(authUserId);
        if (!cancelled) {
          setDailyCommuteSuggestion(data?.suggestion || null);
        }
      } catch (_error) {
        if (!cancelled) {
          setDailyCommuteSuggestion(null);
        }
      }
    };

    loadSuggestion();
    return () => {
      cancelled = true;
    };
  }, [authUserId]);

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

  const handleLocationSelect = (location, type) => {
    const isCoords = typeof location === 'object' && location !== null && typeof location.lat === 'number' && typeof location.lng === 'number';
    if (type === 'pickup') {
      setShowPickupSuggestions(false);
      if (isCoords) {
        setPickup({ lat: location.lat, lng: location.lng, address: location.address || location.label });
        setPickupText(location.label || location.address || `${location.lat.toFixed(4)}, ${location.lng.toFixed(4)}`);
        geocodeHydratedRef.current.pickup = true;
      } else {
        setPickupText(location);
        setPickup(null);
        geocodeHydratedRef.current.pickup = false;
      }
    } else if (type === 'dropoff') {
      setShowDropoffSuggestions(false);
      if (isCoords) {
        setDropoff({ lat: location.lat, lng: location.lng, address: location.address || location.label });
        setDropoffText(location.label || location.address || `${location.lat.toFixed(4)}, ${location.lng.toFixed(4)}`);
        geocodeHydratedRef.current.dropoff = true;
      } else {
        setDropoffText(location);
        setDropoff(null);
        geocodeHydratedRef.current.dropoff = false;
      }
    }
  };

  const handleAcceptDailySuggestion = () => {
    if (!dailyCommuteSuggestion) return;
    setPickup({
      ...dailyCommuteSuggestion.pickup,
      address: dailyCommuteSuggestion.pickup_label,
    });
    setDropoff({
      ...dailyCommuteSuggestion.dropoff,
      address: dailyCommuteSuggestion.dropoff_label,
    });
    setPickupText(dailyCommuteSuggestion.pickup_label);
    setDropoffText(dailyCommuteSuggestion.dropoff_label);
    setDailyCommuteSuggestion(null);
    setShowPickupSuggestions(false);
    setShowDropoffSuggestions(false);
  };

  const handleCancelRideConfirm = async () => {
    try {
      if (currentRideId) {
        await api.cancelRide(currentRideId, authUserId);
      }
      handleReset();
      setShowCancelModal(false);
    } catch (error) {
      console.error('Error canceling ride:', error);
      alert('Failed to cancel ride. Please try again.');
    }
  };

  const showPoolingCard = () => {
    if (!pickup || !dropoff) return false;
    const candidates = buildVehicleCandidates(vehicles, pickup, carType);
    const nearbyCount = Math.max(...candidates.map((c) => c.nearbyRequests));
    return nearbyCount >= 2;
  };

  const getNearbyRequestsCount = () => {
    if (!pickup) return 0;
    const candidates = buildVehicleCandidates(vehicles, pickup, carType);
    return Math.max(...candidates.map((c) => c.nearbyRequests));
  };

  return (
    <div className="booking-page">
      <PolicyBattleModal
        isOpen={battleOpen}
        loading={battleLoading}
        battleSnapshot={battleSnapshot}
        countdownSec={battleCountdown}
        onSelectPolicy={resolveBattleChoice}
      />

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
              {dailyCommuteSuggestion && (
                <DailyCommuteSuggestion
                  suggestion={dailyCommuteSuggestion}
                  onAccept={handleAcceptDailySuggestion}
                  onDismiss={() => setDailyCommuteSuggestion(null)}
                />
              )}
              <div className="input-group">
                <div className="input-row input-wrapper">
                  <span className="dot green-dot"></span>
                  <input 
                    type="text" 
                    placeholder="Pickup location" 
                    value={pickupText} 
                    onChange={(e) => setPickupText(e.target.value)}
                    onFocus={() => setShowPickupSuggestions(true)}
                    className="location-input" 
                  />
                  <button className={`pin-btn ${selectingMode === 'pickup' ? 'active' : ''}`} onClick={() => setSelectingMode(selectingMode === 'pickup' ? null : 'pickup')}>
                    {selectingMode === 'pickup' ? '✕' : '📍'}
                  </button>
                  {showPickupSuggestions && (
                    <LocationSuggestions 
                      type="pickup"
                      isOpen={showPickupSuggestions}
                      mapCenter={mapCenter}
                      cursorLocation={mapCursorLocation}
                      lastMapClickLocation={lastMapClickLocation}
                      zoom={mapZoom}
                      onSelectLocation={(loc) => handleLocationSelect(loc, 'pickup')}
                    />
                  )}
                </div>
                <div className="input-divider"></div>
                <div className="input-row input-wrapper">
                  <span className="dot black-dot"></span>
                  <input 
                    type="text" 
                    placeholder="Dropoff location" 
                    value={dropoffText} 
                    onChange={(e) => setDropoffText(e.target.value)}
                    onFocus={() => setShowDropoffSuggestions(true)}
                    className="location-input" 
                  />
                  <button className={`pin-btn ${selectingMode === 'dropoff' ? 'active' : ''}`} onClick={() => setSelectingMode(selectingMode === 'dropoff' ? null : 'dropoff')}>
                    {selectingMode === 'dropoff' ? '✕' : '🏁'}
                  </button>
                  {showDropoffSuggestions && (
                    <LocationSuggestions 
                      type="dropoff"
                      isOpen={showDropoffSuggestions}
                      mapCenter={mapCenter}
                      cursorLocation={mapCursorLocation}
                      lastMapClickLocation={lastMapClickLocation}
                      zoom={mapZoom}
                      onSelectLocation={(loc) => handleLocationSelect(loc, 'dropoff')}
                    />
                  )}
                </div>
              </div>

              {selectingMode && <div className="map-hint">👆 Click on the map to set {selectingMode} location</div>}

              {vehicles.length > 0 && !pickup && !dropoff && (
                <div className="vehicle-status">🚗 {vehicles.filter((v) => v.status === 'idle').length} vehicles available nearby</div>
              )}

              {pickup && dropoff && (
                <>
                  {showPoolingCard() && (
                    <PoolingSuggestion 
                      nearbyRequestsCount={getNearbyRequestsCount()}
                      matchProbability={35 + (getNearbyRequestsCount() * 15)}
                      onCollapse={() => setShowPoolingSuggestion(false)}
                    />
                  )}

                  <CarTypeSelection 
                    selectedType={carType}
                    onSelectType={setCarType}
                  />

                  <div className="ride-options">
                    <div className="ride-option selected">
                      <span>🚗</span>
                      <div><p>RouteMATE X</p><small>{carType === 'mini' ? '2–3' : carType === 'sedan' ? '4' : '6'} seats • {getDistance()}m away</small></div>
                      <span className="fare">₹{getFare()}</span>
                    </div>
                  </div>
                </>
              )}

              <div className="action-buttons">
                <div className="button-group">
                  <button className="book-btn" onClick={handleBookRide} disabled={!pickup || !dropoff || loading}>
                    {loading ? '⏳ Booking...' : `Book ${policy === 'ml' ? '🤖 ML' : policy === 'greedy' ? '📍 Greedy' : '🎲 Random'} Ride`}
                  </button>
                  {(pickup || dropoff) && <button className="reset-btn" onClick={handleReset}>↺</button>}
                </div>
                {pickup && dropoff && (
                  <div className="policy-badge-container">
                    <AIDecisionBadge policy={policy} isActive={true} />
                  </div>
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
              battleSnapshot={battleSnapshot}
              savings={liveSavings}
              riders={sharedRiders}
              progress={rideProgress}
              rideId={currentRideId}
              comparisonData={comparisonData}
              compareLoading={compareLoading}
              onCompare={handleCompareRide}
              onReset={handleReset}
              onCancel={() => setShowCancelModal(true)}
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
            onMapMove={handleMapMove}
            onCursorMove={handleCursorMove}
            onMapZoom={handleMapZoom}
            center={mapCenter}
            routeGeometry={routeGeometry}
            sharedRiders={sharedRiders}
            liveVehiclePosition={liveVehiclePosition}
          />
        </div>
      </div>

      <CancelRideModal
        isOpen={showCancelModal}
        status={tripStatus}
        onConfirm={handleCancelRideConfirm}
        onCancel={() => setShowCancelModal(false)}
      />
    </div>
  );
}

export default BookingPage;
