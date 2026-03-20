import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const API = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// ==================== SIMULATION & MODEL ====================

export async function runSimulation({
  policy = 'greedy',
  numSteps = 100,
  numVehicles = 5,
  vehicleCapacity = 4,
  requestRate = 2.0,
  modelName = 'ppo_routemate_final',
} = {}) {
  const { data } = await API.post('/simulate', {
    policy,
    num_steps: numSteps,
    num_vehicles: numVehicles,
    vehicle_capacity: vehicleCapacity,
    request_rate: requestRate,
    model_name: modelName,
  });
  return data;
}

export async function getPrediction(observation, modelName = 'ppo_routemate_final') {
  const { data } = await API.post('/predict', {
    observation,
    model_name: modelName,
  });
  return data;
}

export async function getModels() {
  const { data } = await API.get('/models');
  return data;
}

export async function getMetrics() {
  const { data } = await API.get('/metrics');
  return data;
}

// ==================== HEALTH & POLICY ====================

export async function healthCheck() {
  try {
    const { data } = await axios.get('http://localhost:8000/');
    return data;
  } catch (error) {
    console.error('Backend not reachable:', error);
    return { status: 'offline' };
  }
}

export async function getCurrentPolicy() {
  const { data } = await API.get('/policy/current');
  return data;
}

export async function switchPolicy(policy) {
  const { data } = await API.post(`/policy/switch?policy=${policy}`);
  return data;
}

// ==================== VEHICLE & STATS ====================

export async function getVehicles() {
  const { data } = await API.get('/vehicles');
  return data;
}

export async function requestRide(pickup, dropoff, policy = 'greedy') {
  const pickupCoords = pickup?.lat !== undefined ? latLngToGrid(pickup) : pickup;
  const dropoffCoords = dropoff?.lat !== undefined ? latLngToGrid(dropoff) : dropoff;

  const { data } = await API.post('/request-ride', {
    pickup: pickupCoords,
    dropoff: dropoffCoords,
    policy: policy.toLowerCase(),
  });
  return data;
}

export async function getStats() {
  const { data } = await API.get('/stats');
  return data;
}

export async function reloadMLModel() {
  const { data } = await API.post('/model/reload');
  return data;
}

// ==================== MAP UTILITIES ====================

export function latLngToGrid(coords) {
  const LAT_MIN = 18.89;
  const LAT_MAX = 19.27;
  const LNG_MIN = 72.77;
  const LNG_MAX = 73.0;

  const x = Math.floor(((coords.lng - LNG_MIN) / (LNG_MAX - LNG_MIN)) * 10);
  const y = Math.floor(((coords.lat - LAT_MIN) / (LAT_MAX - LAT_MIN)) * 10);

  return {
    x: Math.max(0, Math.min(9, x)),
    y: Math.max(0, Math.min(9, y)),
  };
}

export function gridToLatLng(gridCoords) {
  const LAT_MIN = 18.89;
  const LAT_MAX = 19.27;
  const LNG_MIN = 72.77;
  const LNG_MAX = 73.0;

  return {
    lat: LAT_MIN + (gridCoords.y / 10) * (LAT_MAX - LAT_MIN),
    lng: LNG_MIN + (gridCoords.x / 10) * (LNG_MAX - LNG_MIN),
  };
}

export function formatVehicleForMap(vehicle) {
  const latLng = gridToLatLng(vehicle.location);
  return {
    id: vehicle.vehicle_id,
    position: latLng,
    occupancy: vehicle.occupancy,
    capacity: vehicle.capacity,
    available: vehicle.is_idle,
    icon: vehicle.is_idle ? 'car' : 'taxi',
  };
}

const api = {
  runSimulation,
  getPrediction,
  getModels,
  getMetrics,
  healthCheck,
  getCurrentPolicy,
  getVehicles,
  requestRide,
  switchPolicy,
  getStats,
  reloadMLModel,
  latLngToGrid,
  gridToLatLng,
  formatVehicleForMap,
};

export default api;
