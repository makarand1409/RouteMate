import React, { useEffect, useRef, useState } from 'react';
import './LocationSuggestions.css';
import api from '../services/api';

const DEFAULT_FREQUENT = ['Home', 'Office'];

function formatLatLng(point) {
  return `${point.lat.toFixed(4)}, ${point.lng.toFixed(4)}`;
}

function LocationSuggestions({ type, onSelectLocation, isOpen, mapCenter, cursorLocation, lastMapClickLocation, zoom = 13 }) {
  const [recentLocations, setRecentLocations] = useState([]);
  const [frequentLocations, setFrequentLocations] = useState(DEFAULT_FREQUENT);
  const [placeLabels, setPlaceLabels] = useState({ center: null, cursor: null, clicked: null });
  const [loading, setLoading] = useState({ center: false, cursor: false, clicked: false });
  const cursorTimeout = useRef(null);

  useEffect(() => {
    const stored = localStorage.getItem('locationHistory');
    if (stored) {
      try {
        const history = JSON.parse(stored);
        setRecentLocations(history.recent || []);
        setFrequentLocations(history.frequent?.length ? history.frequent : DEFAULT_FREQUENT);
      } catch (e) {
        console.error('Error loading location history:', e);
      }
    }
  }, []);

  useEffect(() => {
    if (!isOpen) return;

    const fetchPlace = async (key, point) => {
      if (!point) {
        setPlaceLabels((prev) => ({ ...prev, [key]: null }));
        return;
      }
      setLoading((prev) => ({ ...prev, [key]: true }));
      try {
        const response = await api.reverseGeocode(point.lat, point.lng, zoom);
        setPlaceLabels((prev) => ({ ...prev, [key]: response.address || formatLatLng(point) }));
      } catch (_error) {
        setPlaceLabels((prev) => ({ ...prev, [key]: formatLatLng(point) }));
      } finally {
        setLoading((prev) => ({ ...prev, [key]: false }));
      }
    };

    fetchPlace('center', mapCenter);
    fetchPlace('clicked', lastMapClickLocation);
  }, [isOpen, mapCenter, lastMapClickLocation, zoom]);

  useEffect(() => {
    if (!isOpen) {
      if (cursorTimeout.current) {
        clearTimeout(cursorTimeout.current);
        cursorTimeout.current = null;
      }
      return;
    }

    if (!cursorLocation) {
      setPlaceLabels((prev) => ({ ...prev, cursor: null }));
      return;
    }

    if (cursorTimeout.current) {
      clearTimeout(cursorTimeout.current);
    }

    cursorTimeout.current = setTimeout(async () => {
      setLoading((prev) => ({ ...prev, cursor: true }));
      try {
        const response = await api.reverseGeocode(cursorLocation.lat, cursorLocation.lng, zoom);
        setPlaceLabels((prev) => ({ ...prev, cursor: response.address || formatLatLng(cursorLocation) }));
      } catch (_error) {
        setPlaceLabels((prev) => ({ ...prev, cursor: formatLatLng(cursorLocation) }));
      } finally {
        setLoading((prev) => ({ ...prev, cursor: false }));
      }
    }, 350);

    return () => {
      if (cursorTimeout.current) {
        clearTimeout(cursorTimeout.current);
        cursorTimeout.current = null;
      }
    };
  }, [isOpen, cursorLocation, zoom]);

  const persistLocation = (locationLabel) => {
    const stored = localStorage.getItem('locationHistory');
    let history = { recent: [], frequent: DEFAULT_FREQUENT };
    if (stored) {
      try {
        history = JSON.parse(stored);
      } catch (e) {
        // ignore
      }
    }

    history.recent = [locationLabel, ...history.recent.filter((l) => l !== locationLabel)].slice(0, 10);
    localStorage.setItem('locationHistory', JSON.stringify(history));
    setRecentLocations(history.recent);
  };

  const handleSelectLocation = (location) => {
    const label = location.label || location.address || formatLatLng(location);
    persistLocation(label);
    onSelectLocation({ ...location, address: location.address || label, label });
  };

  if (!isOpen) {
    return null;
  }

  return (
    <div className="location-suggestions">
      {(mapCenter || cursorLocation || lastMapClickLocation) && (
        <div className="suggestion-section">
          <h4>🗺️ Map Suggestions</h4>
          {cursorLocation && (
            <button
              className="suggestion-item"
              onClick={() => handleSelectLocation({
                ...cursorLocation,
                address: placeLabels.cursor || `Cursor: ${formatLatLng(cursorLocation)}`,
                label: placeLabels.cursor || `Cursor: ${formatLatLng(cursorLocation)}`,
              })}
            >
              <span className="suggestion-icon">🖱️</span>
              <span className="suggestion-text">
                Use cursor location
                <small>{loading.cursor ? 'Loading nearby place…' : placeLabels.cursor || formatLatLng(cursorLocation)}</small>
              </span>
            </button>
          )}
          {mapCenter && (
            <button
              className="suggestion-item"
              onClick={() => handleSelectLocation({
                ...mapCenter,
                address: placeLabels.center || `Map center: ${formatLatLng(mapCenter)}`,
                label: placeLabels.center || `Map center: ${formatLatLng(mapCenter)}`,
              })}
            >
              <span className="suggestion-icon">📍</span>
              <span className="suggestion-text">
                Use map center
                <small>{loading.center ? 'Loading nearby place…' : placeLabels.center || formatLatLng(mapCenter)}</small>
              </span>
            </button>
          )}
          {lastMapClickLocation && (
            <button
              className="suggestion-item"
              onClick={() => handleSelectLocation({
                ...lastMapClickLocation,
                address: placeLabels.clicked || `Last map point: ${formatLatLng(lastMapClickLocation)}`,
                label: placeLabels.clicked || `Last map point: ${formatLatLng(lastMapClickLocation)}`,
              })}
            >
              <span className="suggestion-icon">📌</span>
              <span className="suggestion-text">
                Use last selected point
                <small>{loading.clicked ? 'Loading nearby place…' : placeLabels.clicked || formatLatLng(lastMapClickLocation)}</small>
              </span>
            </button>
          )}
        </div>
      )}

      {frequentLocations.length > 0 && (
        <div className="suggestion-section">
          <h4>📍 My Places</h4>
          {frequentLocations.map((loc, idx) => (
            <button key={`freq-${idx}`} className="suggestion-item" onClick={() => handleSelectLocation({ label: loc })}>
              <span className="suggestion-icon">🏠</span>
              <span className="suggestion-text">{loc}</span>
            </button>
          ))}
        </div>
      )}

      {recentLocations.length > 0 && (
        <div className="suggestion-section">
          <h4>🕐 Recent Locations</h4>
          {recentLocations.map((loc, idx) => (
            <button key={`recent-${idx}`} className="suggestion-item" onClick={() => handleSelectLocation({ label: loc })}>
              <span className="suggestion-icon">📌</span>
              <span className="suggestion-text">{loc}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default LocationSuggestions;
