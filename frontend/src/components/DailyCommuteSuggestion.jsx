import React from 'react';
import './DailyCommuteSuggestion.css';

function DailyCommuteSuggestion({ suggestion, onAccept, onDismiss }) {
  if (!suggestion) return null;

  const { pickupText, dropoffText, hour, partnerCount, userRouteCount } = suggestion;
  const hasPartner = partnerCount > 0;

  return (
    <div className="daily-commute-card">
      <div className="daily-commute-icon">🗓️</div>
      <div className="daily-commute-content">
        <div className="daily-commute-label">Daily commute suggestion</div>
        <div className="daily-commute-text">
          We found a repeated route from <strong>{pickupText}</strong> to <strong>{dropoffText}</strong> around <strong>{hour}</strong>.
        </div>
        <div className="daily-commute-meta">
          {userRouteCount >= 2 ? (
            <span>{userRouteCount} similar rides logged in your history.</span>
          ) : (
            <span>Frequent route detected in your ride history.</span>
          )}
          {hasPartner && <span>{partnerCount} potential pooling partner{partnerCount > 1 ? 's' : ''} found.</span>}
        </div>
      </div>
      <div className="daily-commute-actions">
        <button className="daily-commute-accept" onClick={onAccept}>Accept suggestion</button>
        <button className="daily-commute-dismiss" onClick={onDismiss}>Maybe later</button>
      </div>
    </div>
  );
}

export default DailyCommuteSuggestion;
