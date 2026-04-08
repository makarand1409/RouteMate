import React, { useState } from 'react';
import './CancelRideModal.css';

function CancelRideModal({ isOpen, status, onConfirm, onCancel }) {
  const [isConfirming, setIsConfirming] = useState(false);

  if (!isOpen) {
    return null;
  }

  const isEnRoute = status === 'picking_up' || status === 'in_progress';
  const title = isEnRoute ? '⚠️  Driver is on the way' : '❌ Cancel Ride';
  const message = isEnRoute
    ? 'Driver is already en route. Are you sure you want to cancel? A cancellation fee may apply.'
    : 'Are you sure you want to cancel this ride?';

  const handleConfirm = async () => {
    setIsConfirming(true);
    try {
      await onConfirm();
    } finally {
      setIsConfirming(false);
    }
  };

  return (
    <div className="cancel-ride-overlay">
      <div className="cancel-ride-modal">
        <h2>{title}</h2>
        <p className="cancel-message">{message}</p>

        {isEnRoute && (
          <div className="cancel-warning">
            <span>⏱️</span>
            <div>
              <strong>Driver will be notified</strong>
              <p>The driver will see your cancellation immediately.</p>
            </div>
          </div>
        )}

        <div className="cancel-actions">
          <button className="cancel-btn-cancel" onClick={onCancel} disabled={isConfirming}>
            Keep Ride
          </button>
          <button
            className="cancel-btn-confirm"
            onClick={handleConfirm}
            disabled={isConfirming}
          >
            {isConfirming ? '⏳ Canceling...' : 'Confirm Cancel'}
          </button>
        </div>

        <p className="cancel-note">
          <small>
            💡 Tip: You can request another ride once cancellation is complete.
          </small>
        </p>
      </div>
    </div>
  );
}

export default CancelRideModal;
