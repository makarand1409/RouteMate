import React, { useState, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import MetricsPanel from '../components/MetricsPanel';
import PolicyComparison from '../components/PolicyComparison';
import RouteAnalysisMap from '../components/RouteAnalysisMap';
import { runSimulation } from '../services/api';
import './DashboardPage.css';

const POLICIES = [
  { key: 'greedy', label: 'Greedy', icon: '📍', desc: 'Nearest vehicle' },
  { key: 'random', label: 'Random', icon: '🎲', desc: 'Random assignment' },
  { key: 'ml',     label: 'ML / PPO', icon: '🤖', desc: 'Trained RL agent' },
];

function DashboardPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuth();
  const routeContext = location.state?.routeContext || null;

  // Simulation controls
  const [policy, setPolicy] = useState('greedy');
  const numSteps = 120;
  const numVehicles = 5;
  const requestRate = 2.0;

  // Simulation results
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Multi-policy comparison history
  const [history, setHistory] = useState([]);

  const handleRun = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await runSimulation({
        policy,
        numSteps,
        numVehicles,
        requestRate,
      });
      setResults(data);
      setHistory(prev => [...prev, data]);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  }, [policy, numSteps, numVehicles, requestRate]);

  const handleRunAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const runs = await Promise.all(
        ['greedy', 'random', 'ml'].map(p =>
          runSimulation({ policy: p, numSteps, numVehicles, requestRate })
        )
      );
      setResults(runs[0]);
      setHistory(prev => [...prev, ...runs]);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  }, [numSteps, numVehicles, requestRate]);

  return (
    <div className="dashboard">
      {/* Nav */}
      <nav className="dash-nav">
        <div className="dash-nav-left">
          <span className="dash-logo" onClick={() => navigate('/')}>RouteMATE</span>
          <span className="dash-badge">Dashboard</span>
        </div>
        <div className="dash-nav-right">
          <button className="dash-nav-btn" onClick={() => navigate('/book')}>
            🚗 Book a Ride
          </button>
          {user && <span className="dash-user">👤 {user.name}</span>}
          <button className="dash-nav-btn" onClick={() => { logout(); navigate('/'); }}>
            Logout
          </button>
        </div>
      </nav>

      <div className="dash-body">
        {/* ─── Controls Panel ─── */}
        <aside className="dash-sidebar">
          <h2>Simulation Controls</h2>

          {/* Policy selector */}
          <div className="ctrl-group">
            <label>Assignment Policy</label>
            <div className="policy-cards">
              {POLICIES.map(p => (
                <div
                  key={p.key}
                  className={`pol-card ${policy === p.key ? 'active' : ''}`}
                  onClick={() => setPolicy(p.key)}
                >
                  <span className="pol-icon">{p.icon}</span>
                  <span className="pol-label">{p.label}</span>
                  <span className="pol-desc">{p.desc}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Actions */}
          <button className="run-btn" onClick={handleRun} disabled={loading}>
            {loading ? '⏳ Running…' : '▶ Run Simulation'}
          </button>
          <button className="run-all-btn" onClick={handleRunAll} disabled={loading}>
            ⚡ Compare All 3 Policies
          </button>

          {error && <div className="dash-error">❌ {error}</div>}
        </aside>

        {/* ─── Main content ─── */}
        <main className="dash-main">
          {!results && !loading && !routeContext && (
            <div className="dash-empty">
              <span>🚀</span>
              <h3>Run a simulation to see results</h3>
              <p>Configure parameters on the left and click "Run Simulation"</p>
            </div>
          )}

          {routeContext && (
            <>
              <RouteAnalysisMap
                routeGeometry={routeContext.routeGeometry}
                pickup={routeContext.pickup}
                dropoff={routeContext.dropoff}
                vehicleLocation={routeContext.vehicleLocation}
              />

              {routeContext.comparison && (
                <div className="comparison-wrapper">
                  <h3>📈 ML vs Greedy (This Completed Route)</h3>
                  <div className="ride-analysis-grid">
                    <div className="ride-analysis-card">
                      <h4>Greedy</h4>
                      <p>Cost: ₹{routeContext.comparison.greedy.estimated_cost}</p>
                      <p>Distance: {routeContext.comparison.greedy.distance_km} km</p>
                      <p>Duration: {routeContext.comparison.greedy.duration_min} min</p>
                    </div>
                    <div className="ride-analysis-card">
                      <h4>ML</h4>
                      <p>Cost: ₹{routeContext.comparison.rl.estimated_cost}</p>
                      <p>Distance: {routeContext.comparison.rl.distance_km} km</p>
                      <p>Duration: {routeContext.comparison.rl.duration_min} min</p>
                    </div>
                    <div className="ride-analysis-card">
                      <h4>Outcome</h4>
                      <p>Winner: {String(routeContext.comparison.winner || '').toUpperCase()}</p>
                      <p>Cost Gain: {routeContext.comparison.improvement.cost_percent}%</p>
                      <p>Time Gain: {routeContext.comparison.improvement.time_percent}%</p>
                      <p>Distance Gain: {routeContext.comparison.improvement.distance_percent}%</p>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}

          {results && (
            <>
              {/* KPI Cards */}
              <MetricsPanel result={results} />

              {/* Policy comparison (if multiple runs exist) */}
              {history.length > 1 && (
                <PolicyComparison history={history} />
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}

export default DashboardPage;
