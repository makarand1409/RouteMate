import React from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

const pickupIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
});

const dropoffIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
});

const vehicleIcon = new L.DivIcon({
  html: '🚕',
  className: 'vehicle-icon',
  iconSize: [30, 30],
  iconAnchor: [15, 15],
});

function RouteAnalysisMap({ routeGeometry = [], pickup, dropoff, vehicleLocation }) {
  const route = routeGeometry
    .filter((p) => Array.isArray(p) && p.length === 2)
    .map((p) => [p[0], p[1]]);

  const center = pickup
    ? [pickup.lat, pickup.lng]
    : route.length
      ? route[0]
      : [19.076, 72.8777];

  return (
    <div className="route-map-wrapper">
      <h3>🗺️ Completed Route Playback</h3>
      <div className="route-map-box">
        <MapContainer center={center} zoom={13} style={{ height: '100%', width: '100%' }}>
          <TileLayer attribution='&copy; OpenStreetMap contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

          {pickup && (
            <Marker position={[pickup.lat, pickup.lng]} icon={pickupIcon}>
              <Popup>Pickup</Popup>
            </Marker>
          )}

          {dropoff && (
            <Marker position={[dropoff.lat, dropoff.lng]} icon={dropoffIcon}>
              <Popup>Dropoff</Popup>
            </Marker>
          )}

          {vehicleLocation && (
            <Marker position={[vehicleLocation.lat, vehicleLocation.lng]} icon={vehicleIcon}>
              <Popup>Vehicle position</Popup>
            </Marker>
          )}

          {route.length > 1 && (
            <Polyline positions={route} pathOptions={{ color: '#111827', weight: 5, opacity: 0.8 }} />
          )}
        </MapContainer>
      </div>
    </div>
  );
}

export default RouteAnalysisMap;
