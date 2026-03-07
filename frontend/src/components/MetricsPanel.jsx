import React from 'react';

function MetricsPanel({ result }) {
  if (!result) return null;

  const kpis = [
    { label: 'Policy', value: result.policy.toUpperCase(), sub: `${result.num_steps} steps` },
    { label: 'Total Requests', value: result.total_requests },
    { label: 'Completed', value: result.completed_requests },
    {
      label: 'Completion Rate',
      value: `${(result.completion_rate * 100).toFixed(1)}%`,
      sub: result.completion_rate >= 0.25 ? '✅ Good' : '⚠️ Low',
    },
    { label: 'Avg Wait', value: `${result.avg_wait_time}s` },
    { label: 'Total Distance', value: result.total_distance_traveled },
  ];

  return (
    <div className="kpi-row">
      {kpis.map((k, i) => (
        <div className="kpi-card" key={i}>
          <div className="kpi-label">{k.label}</div>
          <div className="kpi-value">{k.value}</div>
          {k.sub && <div className="kpi-sub">{k.sub}</div>}
        </div>
      ))}
    </div>
  );
}

export default MetricsPanel;
