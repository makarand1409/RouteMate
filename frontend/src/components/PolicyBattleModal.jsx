import React from 'react';
import PolicyComparisonTable from './PolicyComparisonTable';
import MLReasoningCard from './MLReasoningCard';
import './PolicyBattleModal.css';

function PolicyBattleModal({ isOpen, loading, battleSnapshot, countdownSec = 0, onSelectPolicy }) {
  if (!isOpen) return null;

  return (
    <div className="battle-modal-overlay">
      <div className="battle-modal-card">
        <h3>Choosing Best Driver...</h3>
        <p className="battle-core-message">
          ML does not always choose the nearest car. It chooses the better fleet decision.
        </p>

        {loading ? (
          <div className="battle-loading-wrap">
            <div className="battle-spinner" />
            <p>Running live policy battle (Greedy vs Random vs ML)</p>
          </div>
        ) : (
          <>
            <PolicyComparisonTable policyRows={battleSnapshot?.policyRows || []} />
            <MLReasoningCard battleSnapshot={battleSnapshot} />
            <div className="battle-winner-line">{battleSnapshot?.winnerLine || ''}</div>

            <div className="battle-actions">
              <button className="battle-action-btn" onClick={() => onSelectPolicy && onSelectPolicy('greedy')}>
                Use Greedy
              </button>
              <button className="battle-action-btn" onClick={() => onSelectPolicy && onSelectPolicy('random')}>
                Use Random
              </button>
              <button className="battle-action-btn primary" onClick={() => onSelectPolicy && onSelectPolicy('ml')}>
                Use ML
              </button>
            </div>

            <div className="battle-auto-note">
              Auto-continue with {String(battleSnapshot?.finalPolicy || 'ml').toUpperCase()} in {countdownSec}s
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default PolicyBattleModal;
