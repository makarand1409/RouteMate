import React from 'react';
import './CarTypeSelection.css';

const CAR_TYPES = [
  {
    id: 'mini',
    label: 'Mini',
    icon: '🚗',
    seats: '2–3 seats',
    description: 'Compact & quick',
    price: 'economy',
    badge: '💰',
  },
  {
    id: 'sedan',
    label: 'Sedan',
    icon: '🚕',
    seats: '4 seats',
    description: 'Comfort & style',
    price: 'standard',
    badge: '⭐',
    recommended: true,
  },
  {
    id: 'suv',
    label: 'SUV',
    icon: '🚙',
    seats: '6 seats',
    description: 'Spacious & premium',
    price: 'premium',
    badge: '👑',
  },
];

function CarTypeSelection({ selectedType, onSelectType }) {
  return (
    <div className="car-type-selection">
      <h3>Choose Your Ride Type</h3>
      <div className="car-types-grid">
        {CAR_TYPES.map((car) => (
          <button
            key={car.id}
            className={`car-type-card ${selectedType === car.id ? 'selected' : ''}`}
            onClick={() => onSelectType(car.id)}
          >
            {car.recommended && <div className="recommended-badge">Recommended</div>}
            <div className="car-icon">{car.icon}</div>
            <h4>{car.label}</h4>
            <p className="car-seats">{car.seats}</p>
            <p className="car-description">{car.description}</p>
            <div className="car-badge">{car.badge}</div>
          </button>
        ))}
      </div>
    </div>
  );
}

export default CarTypeSelection;
