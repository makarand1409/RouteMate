import React from 'react';

function MLReasoningCard({ battleSnapshot }) {
  if (!battleSnapshot) return null;

  const tradeoff = battleSnapshot?.reasoning?.tradeoff || 'No tradeoff detected';
  const positives = battleSnapshot?.reasoning?.positives || [];

  return (
    <div className="reasoning-card">
      <h4>Why ML chose this vehicle</h4>

      <div className="reasoning-list">
        {positives.map((item) => (
          <div key={item} className="reasoning-positive">+ {item}</div>
        ))}
      </div>

      <div className="reasoning-tradeoff">- {tradeoff}</div>

      <div className="reasoning-meta">
        <div>Confidence: {battleSnapshot.confidence}%</div>
        <div>
          {battleSnapshot.fallback
            ? 'Fallback triggered: ML ETA too high, using Greedy'
            : 'Fallback: Not triggered'}
        </div>
        <div>Pooling Probability: {battleSnapshot.poolingProbability}%</div>
        <div>Expected Occupancy Gain: +{battleSnapshot.expectedOccupancyGain.toFixed(1)} riders</div>
      </div>
    </div>
  );
}

export default MLReasoningCard;
