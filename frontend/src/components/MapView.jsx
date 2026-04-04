import React from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, CircleMarker, useMapEvents } from 'react-leaflet';
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
  html: '🚗',
  className: 'vehicle-icon',
  iconSize: [30, 30],
  iconAnchor: [15, 15],
});

const assignedVehicleIcon = new L.DivIcon({
  html: '🚕',
  className: 'vehicle-icon',
  iconSize: [35, 35],
  iconAnchor: [17, 17],
});

const sharedRiderIcon = new L.DivIcon({
  html: '<div style="background:#2563eb;color:#fff;border-radius:999px;width:20px;height:20px;display:flex;align-items:center;justify-content:center;font-size:12px;">👤</div>',
  className: 'shared-rider-icon',
  iconSize: [20, 20],
  iconAnchor: [10, 10],
});

function MapClickHandler({ onMapClick, selectingMode }) {
  useMapEvents({
    click: (e) => {
      if (selectingMode) onMapClick(e.latlng);
    },
  });
  return null;
}

function MapView({
  pickup,
  dropoff,
  vehicles = [],
  assignedVehicle,
  onMapClick,
  selectingMode,
  center: centerProp,
  routeGeometry = [],
  sharedRiders = [],
  liveVehiclePosition = null,
}) {
  const center = centerProp || [19.076, 72.8777];

  const assignedVehiclePosition = liveVehiclePosition
    ? [liveVehiclePosition.lat, liveVehiclePosition.lng]
    : (assignedVehicle?.position || null);

  const routePolyline = routeGeometry
    .filter((p) => Array.isArray(p) && p.length === 2)
    .map((p) => [p[0], p[1]]);

  return (
    <MapContainer center={center} zoom={13} style={{ height: '100%', width: '100%' }} className={selectingMode ? 'map-selecting' : ''}>
      <TileLayer attribution='&copy; OpenStreetMap contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

      <MapClickHandler onMapClick={onMapClick} selectingMode={selectingMode} />

      {pickup && (
        <Marker position={[pickup.lat, pickup.lng]} icon={pickupIcon}>
          <Popup>📍 Pickup Location</Popup>
        </Marker>
      )}

      {dropoff && (
        <Marker position={[dropoff.lat, dropoff.lng]} icon={dropoffIcon}>
          <Popup>🏁 Dropoff Location</Popup>
        </Marker>
      )}

      {vehicles.filter((vehicle) => vehicle.id !== assignedVehicle?.id).map((vehicle) => (
        <Marker key={vehicle.id} position={vehicle.position} icon={assignedVehicle?.id === vehicle.id ? assignedVehicleIcon : vehicleIcon}>
          <Popup>
            🚗 Vehicle {vehicle.id}<br />
            Status: {assignedVehicle?.id === vehicle.id ? '⭐ Assigned to you!' : 'Available'}
          </Popup>
        </Marker>
      ))}

      {routePolyline.length > 1 && (
        <Polyline positions={routePolyline} pathOptions={{ color: '#111827', weight: 5, opacity: 0.7 }} />
      )}

      {assignedVehiclePosition && assignedVehicle && (
        <Marker position={assignedVehiclePosition} icon={assignedVehicleIcon}>
          <Popup>
            🚕 Shared Vehicle {assignedVehicle.id}<br />
            Live position update
          </Popup>
        </Marker>
      )}

      {sharedRiders.map((rider) => {
        const p = rider.status === 'completed'
          ? rider.dropoff
          : (rider.status === 'onboard' && liveVehiclePosition ? liveVehiclePosition : rider.pickup);
        if (!p) return null;

        return (
          <Marker key={`rider-${rider.user_id}`} position={[p.lat, p.lng]} icon={sharedRiderIcon}>
            <Popup>
              {rider.name || rider.user_id}<br />
              Status: {rider.status}
            </Popup>
          </Marker>
        );
      })}

      {routePolyline.length > 0 && (
        <CircleMarker center={routePolyline[0]} radius={6} pathOptions={{ color: '#16a34a', fillColor: '#16a34a', fillOpacity: 1 }} />
      )}
    </MapContainer>
  );
}

export default MapView;
