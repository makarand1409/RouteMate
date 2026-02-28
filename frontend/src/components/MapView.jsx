import React, { useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix default marker icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

// Custom Icons
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

// Component to handle map clicks
function MapClickHandler({ onMapClick, selectingMode }) {
  useMapEvents({
    click: (e) => {
      if (selectingMode) {
        onMapClick(e.latlng);
      }
    },
  });
  return null;
}
function MapView({ pickup, dropoff, vehicles, assignedVehicle, onMapClick, selectingMode, center: centerProp }){
  // Center on Bangalore by default (or any city you want)
const center = centerProp || [19.0760, 72.8777];

  return (
    <MapContainer
      center={center}
      zoom={14}
      style={{ height: '100%', width: '100%' }}
      className={selectingMode ? 'map-selecting' : ''}
    >
      <TileLayer
        attribution='&copy; OpenStreetMap contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      <MapClickHandler
        onMapClick={onMapClick}
        selectingMode={selectingMode}
      />

      {/* Pickup Marker */}
      {pickup && (
        <Marker position={[pickup.lat, pickup.lng]} icon={pickupIcon}>
          <Popup>📍 Pickup Location</Popup>
        </Marker>
      )}

      {/* Dropoff Marker */}
      {dropoff && (
        <Marker position={[dropoff.lat, dropoff.lng]} icon={dropoffIcon}>
          <Popup>🏁 Dropoff Location</Popup>
        </Marker>
      )}

      {/* All Vehicles */}
      {vehicles.map((vehicle) => (
        <Marker
          key={vehicle.id}
          position={vehicle.position}
          icon={assignedVehicle?.id === vehicle.id ? assignedVehicleIcon : vehicleIcon}
        >
          <Popup>
            🚗 Vehicle {vehicle.id}<br />
            Status: {assignedVehicle?.id === vehicle.id ? '⭐ Assigned to you!' : 'Available'}
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}

export default MapView;