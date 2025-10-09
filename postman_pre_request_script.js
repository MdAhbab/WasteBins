// Postman Pre-request Script for API readings/submit endpoint
// This script generates realistic sensor data for testing

// Generate random sensor data
const generateSensorReading = () => {
    return {
        node_id: Math.floor(Math.random() * 5) + 1, // Random node ID between 1-5
        temperature: (Math.random() * 20 + 15).toFixed(2), // Temperature between 15-35°C
        humidity: (Math.random() * 40 + 30).toFixed(2), // Humidity between 30-70%
        gas_level: (Math.random() * 1.0).toFixed(3), // Gas level between 0.000-1.000
        waste_level: (Math.random() * 1.0).toFixed(3), // Waste level between 0.000-1.000
        distance_to_next_bin: (Math.random() * 1000 + 100).toFixed(2) // Distance between 100-1100 meters
    };
};

// Generate the sensor reading data
const sensorData = generateSensorReading();

// Set environment variables for the request body
pm.environment.set("node_id", sensorData.node_id);
pm.environment.set("temperature", sensorData.temperature);
pm.environment.set("humidity", sensorData.humidity);
pm.environment.set("gas_level", sensorData.gas_level);
pm.environment.set("waste_level", sensorData.waste_level);
pm.environment.set("distance_to_next_bin", sensorData.distance_to_next_bin);

// Log the generated data for debugging
console.log("Generated sensor reading data:", sensorData);

// Set request headers
pm.request.headers.add({
    key: 'Content-Type',
    value: 'application/json'
});

// Construct the request body
const requestBody = {
    node_id: parseInt(sensorData.node_id),
    temperature: parseFloat(sensorData.temperature),
    humidity: parseFloat(sensorData.humidity),
    gas_level: parseFloat(sensorData.gas_level),
    waste_level: parseFloat(sensorData.waste_level),
    distance_to_next_bin: parseFloat(sensorData.distance_to_next_bin)
};

// Set the request body
pm.request.body.raw = JSON.stringify(requestBody);

console.log("Request body set to:", JSON.stringify(requestBody, null, 2));
