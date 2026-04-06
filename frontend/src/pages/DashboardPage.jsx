import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import RouteAnalysisMap from '../components/RouteAnalysisMap';
import ReplayModal from '../components/ReplayModal';
import './DashboardPage.css';

function DashboardPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuth();
  const routeContext = location.state?.routeContext || null;
  const [replayOpen, setReplayOpen] = useState(false);

  return (
    <div className="dashboard">
      <ReplayModal
        isOpen={replayOpen}
        onClose={() => setReplayOpen(false)}
        battleSnapshot={routeContext?.battleSnapshot || null}
        comparison={routeContext?.comparison || null}
        ridersServed={routeContext?.ridersServed || 1}
      />

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
        <main className="dash-main">
          {!routeContext && (
            <div className="dash-empty">
              <span>🚀</span>
              <h3>No route context yet</h3>
              <p>Complete a ride and open replay from the booking flow.</p>
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

              {routeContext.battleSnapshot && (
                <div className="comparison-wrapper">
                  <h3>Live Policy Battle Snapshot</h3>
                  <p className="dash-core-message">
                    ML does not always choose the nearest car. It chooses the better fleet decision.
                  </p>

                  <div style={{ marginBottom: 10 }}>
                    <button className="run-btn" style={{ maxWidth: 260 }} onClick={() => setReplayOpen(true)}>
                      ▶ Replay Decisions
                    </button>
                  </div>

                  <div className="ride-analysis-grid">
                    <div className="ride-analysis-card">
                      <h4>Battle Winner</h4>
                      <p>{String(routeContext.battleSnapshot.winner || '').toUpperCase()}</p>
                      <p>Confidence: {routeContext.battleSnapshot.confidence || 0}%</p>
                      <p>
                        Fallback: {routeContext.battleSnapshot.fallback ? 'Triggered' : 'Not triggered'}
                      </p>
                    </div>

                    <div className="ride-analysis-card">
                      <h4>System Impact</h4>
                      <p>Riders served: {routeContext.ridersServed || 1}</p>
                      <p>Pooling probability: {routeContext.battleSnapshot.poolingProbability || 0}%</p>
                      <p>
                        Expected occupancy gain: +
                        {Number(routeContext.battleSnapshot.expectedOccupancyGain || 0).toFixed(1)} riders
                      </p>
                    </div>

                    <div className="ride-analysis-card">
                      <h4>Explanation</h4>
                      <p>
                        ML chose a slightly farther vehicle but enabled pooling, reducing total system distance and
                        improving efficiency.
                      </p>
                      <p>
                        Greedy optimizes immediate pickup (local optimum). ML optimizes overall fleet efficiency
                        (global optimum).
                      </p>
                      <p>{routeContext.systemEfficiencyNote || 'System efficiency improved with policy-aware dispatch.'}</p>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}

export default DashboardPage;
