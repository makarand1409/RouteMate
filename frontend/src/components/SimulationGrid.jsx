import React from 'react';

const GRID_SIZE = 10;
const VEHICLE_EMOJI = ['🚗', '🚕', '🚙', '🚐', '🏎️'];

function SimulationGrid({ vehicles = [] }) {
  // Build a map of (x,y) → vehicle for quick lookup
  const vehicleMap = {};
  vehicles.forEach(v => {
    const key = `${v.location[0]},${v.location[1]}`;
    vehicleMap[key] = v;
  });

  const cells = [];
  for (let y = GRID_SIZE - 1; y >= 0; y--) {
    for (let x = 0; x < GRID_SIZE; x++) {
      const key = `${x},${y}`;
      const veh = vehicleMap[key];
      cells.push(
        <div
          key={key}
          className={`grid-cell${veh ? ' has-vehicle' : ''}`}
          title={veh ? `Vehicle ${veh.vehicle_id} — occ: ${veh.occupancy}` : `(${x}, ${y})`}
        >
          {veh ? VEHICLE_EMOJI[(veh.vehicle_id - 1) % VEHICLE_EMOJI.length] : ''}
        </div>
      );
    }
  }

  return (
    <div className="sim-grid-wrapper">
      <h3>🗺️ 10 × 10 City Grid</h3>
      <div className="sim-grid">{cells}</div>

      {vehicles.length > 0 && (
        <table className="vehicle-table">
          <thead>
            <tr>
              <th>Vehicle</th>
              <th>Location</th>
              <th>Passengers</th>
              <th>Queue</th>
              <th>Distance</th>
              <th>Served</th>
            </tr>
          </thead>
          <tbody>
            {vehicles.map(v => (
              <tr key={v.vehicle_id}>
                <td>{VEHICLE_EMOJI[(v.vehicle_id - 1) % VEHICLE_EMOJI.length]} #{v.vehicle_id}</td>
                <td>({v.location[0]}, {v.location[1]})</td>
                <td>{v.occupancy}</td>
                <td>{v.queue_length}</td>
                <td>{v.total_distance}</td>
                <td>{v.total_served}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default SimulationGrid;
