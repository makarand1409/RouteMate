import React from 'react';
import './AIDecisionBadge.css';

/**
 * AIDecisionBadge: Visual indicator that the system uses AI/ML decision making
 * Reinforces the ML storytelling and provides visual feedback about the system intelligence
 */
function AIDecisionBadge({ policy, isActive = true }) {
  const getPolicyColor = () => {
    switch (policy) {
      case 'ml':
        return 'ml-policy';
      case 'greedy':
        return 'greedy-policy';
      case 'random':
        return 'random-policy';
      default:
        return 'default-policy';
    }
  };

  const getPolicyLabel = () => {
    switch (policy) {
      case 'ml':
        return '⭐ ML/AI Optimized';
      case 'greedy':
        return '⚡ Nearest Vehicle';
      case 'random':
        return '⚪ Random Selection';
      default:
        return 'Policy Unknown';
    }
  };

  return (
    <div className={`ai-decision-badge ${getPolicyColor()} ${isActive ? 'active' : 'inactive'}`}>
      <div className="badge-content">
        <span className="badge-text">{getPolicyLabel()}</span>
        {isActive && <div className="badge-pulse"></div>}
      </div>
      {policy === 'ml' && (
        <div className="badge-tooltip">
          🤖 AI Decision Engine Active
        </div>
      )}
    </div>
  );
}

export default AIDecisionBadge;
