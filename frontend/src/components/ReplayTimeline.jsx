import React from 'react';

const STEP_LABELS = [
  'Ride Requested',
  'Policy Comparison',
  'ML Decision Insight',
  'Route Comparison',
  'Outcome Summary',
];

function ReplayTimeline({ currentStep = 0 }) {
  return (
    <div className="replay-timeline">
      {STEP_LABELS.map((label, index) => {
        const state = index < currentStep ? 'done' : index === currentStep ? 'active' : 'pending';
        return (
          <div key={label} className={`replay-timeline-item ${state}`}>
            <div className="replay-timeline-dot">{index + 1}</div>
            <div className="replay-timeline-label">{label}</div>
          </div>
        );
      })}
    </div>
  );
}

export default ReplayTimeline;
