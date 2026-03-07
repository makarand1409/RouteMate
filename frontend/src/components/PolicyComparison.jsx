import React, { useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid,
} from 'recharts';

function PolicyComparison({ history = [] }) {
  // Aggregate: take the LAST run per policy
  const byPolicy = useMemo(() => {
    const map = {};
    history.forEach(r => { map[r.policy] = r; });
    return Object.values(map);
  }, [history]);

  if (byPolicy.length < 2) return null;

  const completionData = byPolicy.map(r => ({
    policy: r.policy.toUpperCase(),
    'Completion %': +(r.completion_rate * 100).toFixed(1),
  }));

  const waitData = byPolicy.map(r => ({
    policy: r.policy.toUpperCase(),
    'Avg Wait': +r.avg_wait_time,
  }));

  return (
    <div className="comparison-wrapper">
      <h3>📊 Policy Comparison</h3>
      <div className="chart-row">
        <div>
          <p style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Completion Rate (%)</p>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={completionData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="policy" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Legend />
              <Bar
                dataKey="Completion %"
                fill="#000"
                radius={[6, 6, 0, 0]}
                barSize={50}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div>
          <p style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Avg Wait Time (steps)</p>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={waitData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="policy" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar
                dataKey="Avg Wait"
                fill="#6366f1"
                radius={[6, 6, 0, 0]}
                barSize={50}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export default PolicyComparison;
