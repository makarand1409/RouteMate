import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './LandingPage.css';

function LandingPage() {
  const navigate = useNavigate();
  const [pickup, setPickup] = useState('');
  const [dropoff, setDropoff] = useState('');

  const handleSeeRides = () => {
    if (!pickup || !dropoff) {
      alert('Please enter pickup and dropoff locations!');
      return;
    }
    navigate('/book', { state: { pickup, dropoff } });
  };

  return (
    <div className="landing">
      {/* Navbar */}
      <nav className="navbar">
        <div className="nav-left">
          <div className="logo">RouteMATE</div>
          <div className="nav-links">
            <a href="/book">Ride</a>
            <a href="/dashboard">Dashboard</a>
            <a href="#">About</a>
          </div>
        </div>
        <div className="nav-right">
          <a href="/login" className="nav-link">Log in</a>
          <button className="signup-btn" onClick={() => navigate('/signup')}>Sign up</button>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="hero">
        {/* Left Side */}
        <div className="hero-left">
          <p className="city-tag">📍 Mumbai, IN</p>
          <h1 className="hero-title">
            Request a ride,<br />anytime anywhere
          </h1>
          <p className="hero-subtitle">
            🤖 Powered by ML matching — smarter than your average cab app
          </p>

          {/* Booking Card */}
          <div className="booking-card">
            <div className="booking-header">
              <span className="booking-now">🕐 Pickup now</span>
            </div>

            {/* Inputs */}
            <div className="input-group">
              <div className="input-row">
                <span className="input-dot green-dot"></span>
                <input
                  type="text"
                  placeholder="Pickup location"
                  value={pickup}
                  onChange={(e) => setPickup(e.target.value)}
                  className="location-input"
                />
                <button className="locate-btn">➤</button>
              </div>
              <div className="input-divider"></div>
              <div className="input-row">
                <span className="input-dot black-dot"></span>
                <input
                  type="text"
                  placeholder="Dropoff location"
                  value={dropoff}
                  onChange={(e) => setDropoff(e.target.value)}
                  className="location-input"
                />
              </div>
            </div>

            <button className="see-rides-btn" onClick={handleSeeRides}>
              See available rides
            </button>
          </div>

          {/* Promo */}
          <p className="promo-text">
            🏷️ <strong>ML-powered matching</strong> — finds your optimal vehicle instantly
          </p>
        </div>

        {/* Right Side - Illustration */}
        <div className="hero-right">
          <div className="map-preview">
            <div className="map-overlay">
              <div className="floating-card card1">
                <span>🚗</span>
                <div>
                  <p>Vehicle 3</p>
                  <small>2 min away</small>
                </div>
              </div>
              <div className="floating-card card2">
                <span>🤖</span>
                <div>
                  <p>ML Assigned</p>
                  <small>Best match!</small>
                </div>
              </div>
              <div className="floating-card card3">
                <span>⭐</span>
                <div>
                  <p>4.9 Rating</p>
                  <small>Top driver</small>
                </div>
              </div>
              {/* Animated vehicles */}
              <div className="animated-city">
                <div className="city-grid">
                  {Array.from({length: 25}).map((_, i) => (
                    <div key={i} className="grid-cell"></div>
                  ))}
                </div>
                <div className="moving-car car-a">🚗</div>
                <div className="moving-car car-b">🚕</div>
                <div className="moving-car car-c">🚙</div>
                <div className="pickup-pin">📍</div>
                <div className="dropoff-pin">🏁</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Features Strip */}
      <div className="features-strip">
        <div className="feature">
          <span>🤖</span>
          <div>
            <p>ML-Powered Matching</p>
            <small>AI assigns optimal vehicle</small>
          </div>
        </div>
        <div className="feature">
          <span>🚗</span>
          <div>
            <p>Vehicle Pooling</p>
            <small>Share rides, save money</small>
          </div>
        </div>
        <div className="feature">
          <span>⚡</span>
          <div>
            <p>Instant Assignment</p>
            <small>Match in milliseconds</small>
          </div>
        </div>
        <div className="feature">
          <span>📊</span>
          <div>
            <p>Smart Analytics</p>
            <small>Track performance live</small>
          </div>
        </div>
      </div>
    </div>
  );
}

export default LandingPage;