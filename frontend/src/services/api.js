import axios from 'axios';

const API = axios.create({
  baseURL: 'http://localhost:8000/api',
  timeout: 30000,
});

/** Run a simulation with specified policy and parameters */
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

/** Get ML model prediction for a given observation vector */
export async function getPrediction(observation, modelName = 'ppo_routemate_final') {
  const { data } = await API.post('/predict', {
    observation,
    model_name: modelName,
  });
  return data;
}

/** List available trained models */
export async function getModels() {
  const { data } = await API.get('/models');
  return data;
}

/** Get policy comparison metrics */
export async function getMetrics() {
  const { data } = await API.get('/metrics');
  return data;
}

/** Health check */
export async function healthCheck() {
  const { data } = await API.get('/', { baseURL: 'http://localhost:8000' });
  return data;
}

export default API;
