import React from 'react';
import './PoolingSuggestion.css';

/**
 * PoolingSuggestion: ML-aligned pooling insight card
 * Shows when there are nearby requests that can be matched
 */
function PoolingSuggestion({ nearbyRequestsCount, matchProbability, onCollapse }) {
  if (!nearbyRequestsCount || nearbyRequestsCount < 2) {
    return null;
  }

  return (
    <div className="pooling-suggestion">
      <div className="pooling-header">
        <span className="pooling-icon">💡</span>
        <div className="pooling-title">
          <h4>Pooling Opportunity</h4>
          <p className="pooling-subtitle">
            {nearbyRequestsCount} rider{nearbyRequestsCount !== 1 ? 's' : ''} nearby going in similar direction
          </p>
        </div>
        {onCollapse && (
          <button className="pooling-close" onClick={onCollapse} title="Dismiss">
            ✕
          </button>
        )}
      </div>

      <div className="pooling-content">
        <div className="pooling-metric">
          <span className="metric-label">Match Probability:</span>
          <span className="metric-value">{matchProbability}%</span>
        </div>
        <div className="pooling-info">
          <small>
            🤖 <strong>ML Insight:</strong> Our system considers pooling opportunities to optimize fleet efficiency and reduce wait times.
          </small>
        </div>
      </div>

      <div className="pooling-badge">💰 Save ~₹{50 + Math.floor(matchProbability / 10)} with shared ride</div>
    </div>
  );
}

export default PoolingSuggestion;
