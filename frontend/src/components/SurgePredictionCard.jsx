import React, { useState, useEffect, useCallback } from 'react';
import api from '../services/api';
import './SurgePredictionCard.css';

/**
 * SurgePredictionCard — ML-powered surge pricing predictor.
 *
 * Shows:
 *   - Current surge multiplier at pickup
 *   - 30-min forecast chart with animated bars
 *   - Best-time recommendation
 *   - Cheaper alternative pickup zones
 *   - Model confidence + info
 *
 * Props:
 *   pickup  — { lat, lng }
 *   dropoff — { lat, lng }
 *   onSelectAlternatePickup — fn({ lat, lng, address }) called when user taps a cheaper zone
 */
function SurgePredictionCard({ pickup, dropoff, onSelectAlternatePickup }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [collapsed, setCollapsed] = useState(false);

  const fetchSurge = useCallback(async () => {
    if (!pickup?.lat || !pickup?.lng) return;
    setLoading(true);
    try {
      const result = await api.getSurgeSmartAdvice(
        pickup.lat,
        pickup.lng,
        dropoff?.lat || pickup.lat + 0.03,
        dropoff?.lng || pickup.lng + 0.02
      );
      setData(result);
    } catch (_err) {
      // Silently fail — surge is a bonus feature.
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [pickup, dropoff]);

  useEffect(() => {
    fetchSurge();
    // Refresh every 30 seconds.
    const interval = setInterval(fetchSurge, 30000);
    return () => clearInterval(interval);
  }, [fetchSurge]);

  if (!pickup) return null;

  const surgeClass = (s) => {
    if (s <= 1.1) return 'low';
    if (s <= 1.8) return 'medium';
    return 'high';
  };

  const barClass = (s, isBest) => {
    if (isBest) return 'bar-best';
    if (s <= 1.1) return 'bar-low';
    if (s <= 1.8) return 'bar-medium';
    return 'bar-high';
  };

  if (collapsed) {
    return (
      <div className="surge-prediction-card" style={{ padding: '12px 16px', cursor: 'pointer' }} onClick={() => setCollapsed(false)}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 18 }}>📊</span>
            <span style={{ fontSize: 13, fontWeight: 600 }}>Surge Prediction</span>
            {data && (
              <span className={`surge-multiplier-big surge-${surgeClass(data.current_surge)}`} style={{ fontSize: 16 }}>
                {data.current_surge}x
              </span>
            )}
          </div>
          <span style={{ fontSize: 12, color: 'rgba(255,255,255,0.4)' }}>▼ Expand</span>
        </div>
      </div>
    );
  }

  return (
    <div className="surge-prediction-card">
      {/* Header */}
      <div className="surge-header">
        <div className="surge-header-left">
          <div className="surge-icon-wrapper">📊</div>
          <h3>
            Surge Prediction
            <small>ML-powered demand forecast</small>
          </h3>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div className="surge-ml-badge">
            <span className="surge-ml-dot" />
            ML ACTIVE
          </div>
          <button className="surge-collapse-btn" onClick={() => setCollapsed(true)}>▲</button>
        </div>
      </div>

      {loading && !data ? (
        <div className="surge-loading">
          <div className="surge-loading-spinner" />
          Analyzing demand patterns…
        </div>
      ) : data ? (
        <>
          {/* Current Surge */}
          <div className="surge-current-display">
            <div className={`surge-multiplier-big surge-${surgeClass(data.current_surge)}`}>
              {data.current_surge}x
            </div>
            <div className="surge-current-meta">
              <div className="surge-fare">₹{data.current_fare}</div>
              <div className="surge-label">
                {data.current_surge <= 1.0 ? 'No Surge' : `${data.current_surge}x Surge Active`}
                {' · '}{data.trip_distance_km} km trip
              </div>
            </div>
          </div>

          {/* Forecast Chart */}
          {data.forecast && data.forecast.length > 0 && (
            <div className="surge-chart-area">
              <div className="surge-chart-title">
                30-Min Surge Forecast
              </div>
              <div className="surge-chart">
                {/* Current bar */}
                <div className="surge-bar-wrapper">
                  <div className="surge-bar-value">{data.current_surge}x</div>
                  <div
                    className={`surge-bar ${barClass(data.current_surge, false)}`}
                    style={{ height: `${Math.max(15, (data.current_surge / 3.5) * 100)}%` }}
                  />
                  <div className="surge-bar-label">Now</div>
                </div>
                {data.forecast.map((f, i) => {
                  const isBest = data.best_future_time && f.offset_min === data.best_future_time.offset_min;
                  return (
                    <div className="surge-bar-wrapper" key={i}>
                      <div className="surge-bar-value">{f.surge}x</div>
                      <div
                        className={`surge-bar ${barClass(f.surge, isBest)}`}
                        style={{
                          height: `${Math.max(15, (f.surge / 3.5) * 100)}%`,
                          transitionDelay: `${i * 80}ms`,
                        }}
                      />
                      <div className="surge-bar-label">+{f.offset_min}m</div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Recommendation */}
          {data.recommendation && (
            <div className="surge-recommendation">
              <div className="surge-rec-action">
                <div
                  className={`surge-rec-icon ${
                    data.recommendation.strategy === 'wait'
                      ? 'rec-wait'
                      : data.recommendation.strategy === 'walk'
                      ? 'rec-walk'
                      : 'rec-book'
                  }`}
                >
                  {data.recommendation.strategy === 'wait' ? '⏰' : data.recommendation.strategy === 'walk' ? '🚶' : '🚀'}
                </div>
                <div className="surge-rec-text">
                  <strong>
                    {data.recommendation.strategy === 'wait'
                      ? 'Wait for Lower Surge'
                      : data.recommendation.strategy === 'walk'
                      ? 'Walk to Cheaper Zone'
                      : 'Book Now!'}
                  </strong>
                  <span>{data.recommendation.message}</span>
                </div>
              </div>
              {(data.wait_savings > 0 || data.walk_savings > 0) && (
                <div className="surge-savings-badge">
                  💰 Save up to ₹{Math.max(data.wait_savings || 0, data.walk_savings || 0)}
                </div>
              )}
            </div>
          )}

          {/* Cheaper Zones */}
          {data.cheaper_pickup_zones && data.cheaper_pickup_zones.length > 0 && (
            <div className="surge-cheaper-zones">
              <div className="surge-cheaper-title">
                🏃 Cheaper Pickup Alternatives
              </div>
              {data.cheaper_pickup_zones.slice(0, 3).map((zone, i) => (
                <div
                  className="surge-zone-row"
                  key={i}
                  onClick={() =>
                    onSelectAlternatePickup &&
                    onSelectAlternatePickup({
                      lat: zone.lat,
                      lng: zone.lng,
                      address: `Cheaper Zone (${zone.surge}x surge)`,
                    })
                  }
                  style={{ cursor: onSelectAlternatePickup ? 'pointer' : 'default' }}
                >
                  <div className={`surge-zone-surge zone-${surgeClass(zone.surge)}`}>{zone.surge}x</div>
                  <div className="surge-zone-info">
                    🚶 {zone.walk_min} min walk · {zone.walk_km} km
                  </div>
                  <div className="surge-zone-savings">-₹{zone.savings}</div>
                </div>
              ))}
            </div>
          )}

          {/* Model Info */}
          <div className="surge-model-info">
            <span>🤖 {data.ml_model || 'DQN Forecaster'}</span>
            <div className="surge-confidence">
              <span>Confidence</span>
              <div className="confidence-bar-bg">
                <div
                  className="confidence-bar-fill"
                  style={{ width: `${(data.recommendation?.confidence || 0.87) * 100}%` }}
                />
              </div>
              <span>{Math.round((data.recommendation?.confidence || 0.87) * 100)}%</span>
            </div>
          </div>
        </>
      ) : (
        <div className="surge-loading" style={{ opacity: 0.5 }}>
          Surge data unavailable
        </div>
      )}
    </div>
  );
}

export default SurgePredictionCard;
