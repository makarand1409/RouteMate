import React, { useState, useEffect, useCallback } from 'react';
import api from '../services/api';
import './AIExplainerTimeline.css';

/**
 * AIExplainerTimeline — Explainable AI (XAI) component.
 *
 * Visualizes WHY the ML model made each decision for a ride with:
 *   1) Animated decision timeline
 *   2) Feature importance bar chart
 *   3) Neural network visualization (SVG)
 *   4) Counterfactual analysis
 *
 * Props:
 *   rideId   — current ride ID
 *   userId   — current user ID
 *   policy   — policy used ('ml', 'greedy', 'random')
 */
function AIExplainerTimeline({ rideId, userId, policy }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [collapsed, setCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState('timeline');

  const fetchExplanation = useCallback(async () => {
    if (!rideId) return;
    setLoading(true);
    try {
      const result = await api.getRideExplanation(rideId, userId);
      setData(result);
    } catch (_err) {
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [rideId, userId]);

  useEffect(() => {
    fetchExplanation();
  }, [fetchExplanation]);

  if (!rideId) return null;

  if (collapsed) {
    return (
      <div className="xai-explainer xai-collapsed">
        <div className="xai-header" onClick={() => setCollapsed(false)}>
          <div className="xai-header-left">
            <div className="xai-header-icon">🧠</div>
            <h3>
              AI Decision Explainer
              {data && (
                <small>{Math.round((data.overall_confidence || 0.88) * 100)}% confidence</small>
              )}
            </h3>
          </div>
          <div className="xai-header-right">
            <span className="xai-collapse-btn">▼ Expand</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="xai-explainer">
      {/* Header */}
      <div className="xai-header" onClick={() => setCollapsed(true)}>
        <div className="xai-header-left">
          <div className="xai-header-icon">🧠</div>
          <h3>
            AI Decision Explainer
            <small>Explainable AI — see why the model chose this</small>
          </h3>
        </div>
        <div className="xai-header-right">
          <div className="xai-badge">
            <span className="xai-badge-dot" />
            XAI
          </div>
          <button className="xai-collapse-btn" onClick={(e) => { e.stopPropagation(); setCollapsed(true); }}>
            ▲
          </button>
        </div>
      </div>

      {loading && !data ? (
        <div className="xai-loading">
          <div className="xai-loading-spinner" />
          Analyzing AI decision process…
        </div>
      ) : data ? (
        <div className="xai-body">
          {/* Tabs */}
          <div className="xai-tabs">
            <button
              className={`xai-tab ${activeTab === 'timeline' ? 'active' : ''}`}
              onClick={() => setActiveTab('timeline')}
            >
              🕐 Timeline
            </button>
            <button
              className={`xai-tab ${activeTab === 'features' ? 'active' : ''}`}
              onClick={() => setActiveTab('features')}
            >
              📊 Features
            </button>
            <button
              className={`xai-tab ${activeTab === 'network' ? 'active' : ''}`}
              onClick={() => setActiveTab('network')}
            >
              🧬 Network
            </button>
            <button
              className={`xai-tab ${activeTab === 'whatif' ? 'active' : ''}`}
              onClick={() => setActiveTab('whatif')}
            >
              ⚖️ What-If
            </button>
          </div>

          {/* Timeline Tab */}
          {activeTab === 'timeline' && data.decision_timeline && (
            <div className="xai-timeline">
              {data.decision_timeline.map((step, i) => (
                <div
                  className="xai-timeline-step"
                  key={step.step}
                  style={{ animationDelay: `${i * 120}ms` }}
                >
                  <div className={`xai-step-dot type-${step.type}`} />
                  <div className="xai-step-content">
                    <div className="xai-step-title">
                      <span className="step-icon">{step.icon}</span>
                      {step.title}
                    </div>
                    <div className="xai-step-desc">{step.description}</div>
                    {step.confidence != null && (
                      <div className="xai-step-confidence">
                        🎯 {(step.confidence * 100).toFixed(0)}% confidence
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Features Tab */}
          {activeTab === 'features' && data.feature_importance && (
            <div className="xai-features">
              {data.feature_importance.map((feat, i) => {
                const pct = feat.importance;
                const cls = pct > 22 ? 'importance-high' : pct > 12 ? 'importance-medium' : 'importance-low';
                return (
                  <div key={feat.feature}>
                    <div
                      className="xai-feature-row"
                      style={{ animationDelay: `${i * 80}ms` }}
                    >
                      <div className="xai-feature-label">{feat.feature}</div>
                      <div className="xai-feature-bar-bg">
                        <div
                          className={`xai-feature-bar-fill ${cls}`}
                          style={{ width: `${Math.min(100, pct * 3.2)}%` }}
                        />
                      </div>
                      <div className="xai-feature-pct">{pct}%</div>
                    </div>
                    {feat.description && (
                      <div className="xai-feature-tooltip">{feat.description}</div>
                    )}
                  </div>
                );
              })}

              {/* Confidence Ring */}
              <ConfidenceRing confidence={data.overall_confidence || 0.88} modelInfo={data.model_info} />
            </div>
          )}

          {/* Neural Network Tab */}
          {activeTab === 'network' && data.neural_network && (
            <NeuralNetworkViz layers={data.neural_network} modelInfo={data.model_info} />
          )}

          {/* What-If Tab */}
          {activeTab === 'whatif' && data.counterfactual && (
            <CounterfactualView data={data.counterfactual} />
          )}

          {/* Footer */}
          <div className="xai-model-footer">
            <div className="xai-model-badge">
              🤖 {data.model_info?.algorithm || 'DQN'}
            </div>
            <div className="xai-model-tags">
              <span className="xai-model-tag">{data.model_info?.training_steps || '200K'} steps</span>
              <span className="xai-model-tag">{data.model_info?.activation || 'ReLU'}</span>
              <span className="xai-model-tag">{data.model_info?.optimizer || 'Adam'}</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="xai-loading" style={{ opacity: 0.5 }}>
          Explanation unavailable
        </div>
      )}
    </div>
  );
}

/* ========== Confidence Ring Sub-component ========== */
function ConfidenceRing({ confidence, modelInfo }) {
  const pct = Math.round(confidence * 100);
  const radius = 34;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (confidence * circumference);

  return (
    <div className="xai-confidence-ring-area">
      <div className="xai-confidence-ring-wrapper">
        <svg className="xai-confidence-ring-svg" width="80" height="80" viewBox="0 0 80 80">
          <defs>
            <linearGradient id="confGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#7c3aed" />
              <stop offset="100%" stopColor="#06b6d4" />
            </linearGradient>
          </defs>
          <circle className="xai-confidence-ring-bg" cx="40" cy="40" r={radius} />
          <circle
            className="xai-confidence-ring-fill"
            cx="40"
            cy="40"
            r={radius}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="xai-confidence-ring-text">{pct}%</div>
        <div className="xai-confidence-ring-label">Confidence</div>
      </div>
      <div className="xai-confidence-details">
        <div className="xai-confidence-detail-row">
          <strong>Model</strong>
          <span>{modelInfo?.algorithm || 'DQN'}</span>
        </div>
        <div className="xai-confidence-detail-row">
          <strong>Layers</strong>
          <span>{modelInfo?.hidden_layers || 2} hidden</span>
        </div>
        <div className="xai-confidence-detail-row">
          <strong>Trained</strong>
          <span>{modelInfo?.training_steps || '200K'} steps</span>
        </div>
        <div className="xai-confidence-detail-row">
          <strong>Features</strong>
          <span>{modelInfo?.features_used || 7} inputs</span>
        </div>
      </div>
    </div>
  );
}

/* ========== Neural Network Visualization ========== */
function NeuralNetworkViz({ layers, modelInfo }) {
  if (!layers || layers.length === 0) return null;

  const svgW = 360;
  const svgH = 220;
  const layerSpacing = svgW / (layers.length + 1);
  const labelY = svgH - 8;

  // Pre-compute neuron positions.
  const layerPositions = layers.map((layer, li) => {
    const x = (li + 1) * layerSpacing;
    const maxNeurons = Math.min(layer.neurons, 10);
    const neuronSpacing = Math.min(22, (svgH - 50) / maxNeurons);
    const startY = (svgH - 20) / 2 - ((maxNeurons - 1) * neuronSpacing) / 2;
    const neurons = [];
    for (let ni = 0; ni < maxNeurons; ni++) {
      const y = startY + ni * neuronSpacing;
      const activation = layer.activations[ni] || 0;
      neurons.push({ x, y, activation });
    }
    return { x, neurons, name: layer.name, type: layer.type };
  });

  return (
    <div className="xai-neural-net">
      <div className="xai-nn-svg-container">
        <svg className="xai-nn-svg" viewBox={`0 0 ${svgW} ${svgH}`}>
          {/* Connections */}
          {layerPositions.map((layer, li) => {
            if (li === 0) return null;
            const prev = layerPositions[li - 1];
            return prev.neurons.map((pn, pi) =>
              layer.neurons.map((cn, ci) => {
                const strength = (pn.activation + cn.activation) / 2;
                const isActive = strength > 0.45;
                return (
                  <line
                    key={`c-${li}-${pi}-${ci}`}
                    x1={pn.x}
                    y1={pn.y}
                    x2={cn.x}
                    y2={cn.y}
                    className={`xai-connection ${isActive ? 'active' : ''}`}
                    style={{
                      opacity: 0.1 + strength * 0.5,
                      animationDelay: `${(pi + ci) * 100}ms`,
                    }}
                  />
                );
              })
            );
          })}

          {/* Neurons */}
          {layerPositions.map((layer, li) =>
            layer.neurons.map((n, ni) => {
              const r = 5 + n.activation * 4;
              const hue = layer.type === 'input' ? 250 : layer.type === 'output' ? 180 : 270;
              const color = `hsla(${hue}, 70%, ${40 + n.activation * 30}%, ${0.5 + n.activation * 0.5})`;
              return (
                <g key={`n-${li}-${ni}`} className="xai-neuron xai-neuron-glow">
                  <circle
                    cx={n.x}
                    cy={n.y}
                    r={r}
                    fill={color}
                    stroke={`hsla(${hue}, 80%, 70%, 0.6)`}
                    strokeWidth="1"
                    style={{ animationDelay: `${(li * 200) + (ni * 80)}ms` }}
                  />
                </g>
              );
            })
          )}

          {/* Layer labels */}
          {layerPositions.map((layer, li) => (
            <text
              key={`label-${li}`}
              x={layer.x}
              y={labelY}
              className="xai-nn-layer-label"
            >
              {layer.name.replace(' Layer', '')}
            </text>
          ))}
        </svg>
      </div>

      <div className="xai-nn-info">
        <span className="xai-nn-info-tag">12 → 8 → 6 → 4 neurons</span>
        <span className="xai-nn-info-tag">{modelInfo?.activation || 'ReLU'} activation</span>
        <span className="xai-nn-info-tag">{modelInfo?.optimizer || 'Adam'} optimizer</span>
      </div>
    </div>
  );
}

/* ========== Counterfactual What-If View ========== */
function CounterfactualView({ data }) {
  if (!data) return null;
  const { actual, counterfactual, savings, verdict } = data;

  return (
    <div className="xai-counterfactual">
      <div className="xai-cf-comparison">
        {/* Actual */}
        <div className="xai-cf-card actual">
          <div className="xai-cf-label">
            ✅ Actual ({actual.policy.toUpperCase()})
          </div>
          <div className="xai-cf-metric">
            <div className="xai-cf-metric-value">₹{actual.fare}</div>
            <div className="xai-cf-metric-label">Total Fare</div>
          </div>
          <div className="xai-cf-row">
            <span>Distance</span>
            <span>{actual.distance_km} km</span>
          </div>
          <div className="xai-cf-row">
            <span>Duration</span>
            <span>{actual.duration_min} min</span>
          </div>
        </div>

        {/* Counterfactual */}
        <div className="xai-cf-card alternative">
          <div className="xai-cf-label">
            🔮 If {counterfactual.policy.toUpperCase()}
          </div>
          <div className="xai-cf-metric">
            <div className="xai-cf-metric-value">₹{counterfactual.fare}</div>
            <div className="xai-cf-metric-label">Total Fare</div>
          </div>
          <div className="xai-cf-row">
            <span>Distance</span>
            <span>{counterfactual.distance_km} km</span>
          </div>
          <div className="xai-cf-row">
            <span>Duration</span>
            <span>{counterfactual.duration_min} min</span>
          </div>
        </div>
      </div>

      {/* Savings Summary */}
      {(savings.money !== 0 || savings.time_min !== 0) && (
        <div className="xai-cf-comparison">
          <div className="xai-cf-card" style={{ gridColumn: 'span 2' }}>
            <div className="xai-cf-label" style={{ color: '#a78bfa' }}>📈 Difference</div>
            <div style={{ display: 'flex', justifyContent: 'space-around', paddingTop: 4 }}>
              <div>
                <div className="xai-cf-metric-value" style={{ fontSize: 16, color: savings.money > 0 ? '#ef4444' : '#10b981' }}>
                  {savings.money > 0 ? '+' : ''}₹{savings.money}
                </div>
                <div className="xai-cf-metric-label">Fare Diff</div>
              </div>
              <div>
                <div className="xai-cf-metric-value" style={{ fontSize: 16, color: savings.time_min > 0 ? '#ef4444' : '#10b981' }}>
                  {savings.time_min > 0 ? '+' : ''}{savings.time_min} min
                </div>
                <div className="xai-cf-metric-label">Time Diff</div>
              </div>
              <div>
                <div className="xai-cf-metric-value" style={{ fontSize: 16, color: savings.distance_km > 0 ? '#ef4444' : '#10b981' }}>
                  {savings.distance_km > 0 ? '+' : ''}{savings.distance_km} km
                </div>
                <div className="xai-cf-metric-label">Dist Diff</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Verdict */}
      <div className="xai-cf-verdict">
        <span>💡</span>
        {verdict}
      </div>
    </div>
  );
}

export default AIExplainerTimeline;
