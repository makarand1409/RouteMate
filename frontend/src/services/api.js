const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';
const WS_BASE_URL = API_BASE_URL.replace('http://', 'ws://').replace('https://', 'wss://');

function buildQuery(params) {
  return new URLSearchParams(params).toString();
}

async function jsonFetch(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const err = await response.json();
      detail = err.detail || err.message || detail;
    } catch (_e) {
      // ignore
    }
    throw new Error(detail);
  }
  return await response.json();
}

class RouteMATEAPI {
  async healthCheck() {
    try {
      return await jsonFetch('http://localhost:8000/');
    } catch (_error) {
      return { status: 'offline' };
    }
  }

  async getCurrentPolicy() {
    return await jsonFetch(`${API_BASE_URL}/policy/current`);
  }

  async switchPolicy(policy) {
    return await jsonFetch(`${API_BASE_URL}/policy/switch?policy=${encodeURIComponent(policy)}`, {
      method: 'POST',
    });
  }

  async getVehicles() {
    return await jsonFetch(`${API_BASE_URL}/vehicles`);
  }

  async requestRide(pickup, dropoff, policy = 'greedy', userContext = null) {
    const pickupGrid = this.latLngToGrid(pickup);
    const dropoffGrid = this.latLngToGrid(dropoff);

    return await jsonFetch(`${API_BASE_URL}/request-ride`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        pickup: { x: pickupGrid.x, y: pickupGrid.y, lat: pickup.lat, lng: pickup.lng },
        dropoff: { x: dropoffGrid.x, y: dropoffGrid.y, lat: dropoff.lat, lng: dropoff.lng },
        policy,
        user_id: userContext?.userId,
        user_name: userContext?.userName,
        user_email: userContext?.userEmail,
      }),
    });
  }

  async getRideState(rideId) {
    return await jsonFetch(`${API_BASE_URL}/rides/${encodeURIComponent(rideId)}`);
  }

  async getRideComparison(rideId, userId) {
    const query = buildQuery({ user_id: userId });
    return await jsonFetch(`${API_BASE_URL}/rides/${encodeURIComponent(rideId)}/comparison?${query}`);
  }

  async geocodeAddress(address) {
    const query = buildQuery({ address });
    return await jsonFetch(`${API_BASE_URL}/geo/geocode?${query}`);
  }

  async getRoute(pickup, dropoff) {
    const query = buildQuery({
      pickup_lat: String(pickup.lat),
      pickup_lng: String(pickup.lng),
      dropoff_lat: String(dropoff.lat),
      dropoff_lng: String(dropoff.lng),
    });
    return await jsonFetch(`${API_BASE_URL}/geo/route?${query}`);
  }

  getRideWebSocketUrl(userId) {
    return `${WS_BASE_URL.replace('/api', '')}/ws/rides/${userId}`;
  }

  async getStats() {
    return await jsonFetch(`${API_BASE_URL}/stats`);
  }

  async reloadMLModel() {
    return await jsonFetch(`${API_BASE_URL}/model/reload`, { method: 'POST' });
  }

  async runSimulation(params = {}) {
    const {
      policy = 'greedy',
      numSteps = 100,
      numVehicles = 5,
      requestRate = 2.0,
      vehicleCapacity = 4,
    } = params;

    return await jsonFetch(`${API_BASE_URL}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        policy,
        num_steps: numSteps,
        num_vehicles: numVehicles,
        request_rate: requestRate,
        vehicle_capacity: vehicleCapacity,
      }),
    });
  }

  latLngToGrid(coords) {
    const x = Math.floor(((coords.lng - 72.77) / (73.0 - 72.77)) * 10);
    const y = Math.floor(((coords.lat - 18.89) / (19.27 - 18.89)) * 10);
    return { x: Math.max(0, Math.min(9, x)), y: Math.max(0, Math.min(9, y)) };
  }

  gridToLatLng(x, y) {
    return {
      lat: 18.89 + (y / 10) * (19.27 - 18.89),
      lng: 72.77 + (x / 10) * (73.0 - 72.77),
    };
  }

  formatVehicleForMap(vehicle) {
    const x = vehicle.location?.x ?? 0;
    const y = vehicle.location?.y ?? 0;
    const pos = this.gridToLatLng(x, y);
    return {
      id: vehicle.vehicle_id,
      position: [pos.lat, pos.lng],
      occupancy: vehicle.occupancy,
      capacity: vehicle.capacity,
      status: vehicle.is_idle ? 'idle' : 'busy',
      name: `Vehicle ${vehicle.vehicle_id}`,
    };
  }
}

const api = new RouteMATEAPI();
export default api;

export const runSimulation = (...args) => api.runSimulation(...args);
export const healthCheck = (...args) => api.healthCheck(...args);
export const getCurrentPolicy = (...args) => api.getCurrentPolicy(...args);
export const getVehicles = (...args) => api.getVehicles(...args);
export const requestRide = (...args) => api.requestRide(...args);
