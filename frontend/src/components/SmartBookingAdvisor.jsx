import React, { useState, useEffect } from 'react';
import './SmartBookingAdvisor.css';

/**
 * SmartBookingAdvisor — "Wait or Book Now?" card with countdown timer.
 *
 * This component is the quick-glance version of surge intelligence.
 * It shows one clear recommendation with a savings amount and optional
 * countdown for the "wait" strategy.
 *
 * Props:
 *   surgeData   — The response from /api/surge/smart-advice
 *   onDismiss   — fn() called when user closes the card
 *   onBookNow   — fn() called when user wants to book immediately
 *   onWait      — fn() called when user decides to wait
 */
function SmartBookingAdvisor({ surgeData, onDismiss, onBookNow, onWait }) {
  const [countdown, setCountdown] = useState(0);
  const [dismissed, setDismissed] = useState(false);

  const strategy = surgeData?.recommendation?.strategy || 'book_now';
  const message = surgeData?.recommendation?.message || '';
  const waitMinutes = surgeData?.best_future_time?.offset_min || 0;
  const savings = Math.max(surgeData?.wait_savings || 0, surgeData?.walk_savings || 0);

  useEffect(() => {
    if (strategy !== 'wait' || waitMinutes <= 0) return undefined;

    setCountdown(waitMinutes * 60);
    const timer = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(timer);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [strategy, waitMinutes]);

  if (!surgeData || dismissed) return null;

  // Don't show at all if surge is 1.0 (no surge).
  if (surgeData.current_surge <= 1.0 && strategy === 'book_now') return null;

  const formatCountdown = (seconds) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${String(s).padStart(2, '0')}`;
  };

  const emojis = {
    wait: '⏰',
    book_now: '🚀',
    walk: '🚶',
  };

  const titles = {
    wait: 'Smart Wait Advisor',
    book_now: 'Optimal Booking Window',
    walk: 'Cheaper Pickup Nearby',
  };

  const handleDismiss = () => {
    setDismissed(true);
    if (onDismiss) onDismiss();
  };

  return (
    <div className={`smart-advisor strategy-${strategy}`}>
      <button className="advisor-dismiss" onClick={handleDismiss}>✕</button>

      <div className="advisor-content">
        <div className="advisor-emoji">{emojis[strategy] || '💡'}</div>
        <div className="advisor-body">
          <div className="advisor-title">{titles[strategy]}</div>
          <div className="advisor-message">{message}</div>

          <div className="advisor-savings-row">
            {savings > 0 && (
              <div className="advisor-savings-amount">
                💰 Save ₹{savings}
              </div>
            )}
            {strategy === 'wait' && countdown > 0 && (
              <div className="advisor-countdown">
                <span className="advisor-countdown-dot" />
                {formatCountdown(countdown)} remaining
              </div>
            )}
          </div>

          <div className="advisor-cta">
            {strategy === 'wait' ? (
              <>
                <button className="advisor-btn" onClick={onBookNow}>
                  Book Now (₹{surgeData.current_fare})
                </button>
                <button className="advisor-btn advisor-btn-primary" onClick={onWait || onDismiss}>
                  ⏰ Wait & Save
                </button>
              </>
            ) : strategy === 'walk' ? (
              <>
                <button className="advisor-btn" onClick={onBookNow}>
                  Book Here
                </button>
                <button className="advisor-btn advisor-btn-primary" onClick={onWait || onDismiss}>
                  🚶 Walk & Save
                </button>
              </>
            ) : (
              <button className="advisor-btn advisor-btn-primary" onClick={onBookNow} style={{ flex: 1 }}>
                🚀 Book Now — Best Price!
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default SmartBookingAdvisor;
