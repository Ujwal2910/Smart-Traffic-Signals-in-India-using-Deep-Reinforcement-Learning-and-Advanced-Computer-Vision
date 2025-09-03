document.addEventListener('DOMContentLoaded', () => {
    const map = L.map('map').setView([20.2961, 85.8245], 13);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(map);

    fetchRoadNetwork(map);

    // Sidebar navigation
    const sidebarItems = document.querySelectorAll('#sidebar ul li');
    const contentPanels = document.querySelectorAll('.content-panel');

    sidebarItems.forEach((item, index) => {
        item.addEventListener('click', () => {
            // Deactivate current active items
            document.querySelector('#sidebar .active').classList.remove('active');
            document.querySelector('.content-panel.active').classList.remove('active');

            // Activate new items
            item.classList.add('active');
            contentPanels[index].classList.add('active');

            // Special case for map
            if (contentPanels[index].id === 'map') {
                map.invalidateSize();
            }
        });
    });
});

class RoadNetwork {
    constructor() {
        this.nodes = {};
        this.ways = {};
    }

    addNode(node) {
        this.nodes[node.id] = { lat: node.lat, lon: node.lon, ways: [] };
    }

    addWay(way) {
        this.ways[way.id] = { nodes: way.nodes };
        way.nodes.forEach(nodeId => {
            if (this.nodes[nodeId]) {
                this.nodes[nodeId].ways.push(way.id);
            }
        });
    }
}

function fetchRoadNetwork(map) {
    const roadNetwork = new RoadNetwork();
    const bbox = "20.15,85.77,20.35,85.97"; // S,W,N,E for Bhubaneswar
    const query = `
        [out:json];
        (
            way["highway"](${bbox});
        );
        out body;
        >;
        out skel qt;
    `;
    const url = `https://overpass-api.de/api/interpreter?data=${encodeURIComponent(query)}`;

    fetch(url)
        .then(response => response.json())
        .then(data => {
            data.elements.forEach(el => {
                if (el.type === 'node') {
                    roadNetwork.addNode(el);
                }
            });

            data.elements.forEach(el => {
                if (el.type === 'way' && el.nodes) {
                    roadNetwork.addWay(el);
                    const latlngs = el.nodes.map(nodeId => {
                        const node = roadNetwork.nodes[nodeId];
                        return node ? [node.lat, node.lon] : null;
                    }).filter(n => n);

                    if (latlngs.length > 1) {
                        L.polyline(latlngs, { color: 'grey', weight: 3 }).addTo(map);
                    }
                }
            });

            console.log("Road network created, starting simulation...");
            const simulation = new Simulation(map, roadNetwork);
            simulation.start(50); // Start simulation with 50 vehicles

            const greenWave = new GreenWave(map, roadNetwork, simulation);

            const eventHorizon = new EventHorizon(map);

            const analytics = new Analytics(simulation);

            const weather = new Weather();
        })
        .catch(error => {
            console.error('Error fetching road network:', error);
        });
}

class GreenWave {
    constructor(map, roadNetwork, simulation) {
        this.map = map;
        this.roadNetwork = roadNetwork;
        this.simulation = simulation;
        this.selectingStart = false;
        this.selectingEnd = false;
        this.startNode = null;
        this.endNode = null;
        this.startMarker = null;
        this.endMarker = null;

        this.selectStartBtn = document.getElementById('select-start-btn');
        this.selectEndBtn = document.getElementById('select-end-btn');
        this.startGwBtn = document.getElementById('start-gw-btn');
        this.cancelGwBtn = document.getElementById('cancel-gw-btn');
        this.startCoordsEl = document.getElementById('start-coords');
        this.endCoordsEl = document.getElementById('end-coords');

        this.path = null;

        this._bindEvents();
    }

    _bindEvents() {
        this.startGwBtn.addEventListener('click', () => {
            if (this.startNode && this.endNode) {
                this.path = this._aStarSearch(this.startNode.id, this.endNode.id);
                if (this.path) {
                    this._visualizePath(this.path);
                    this.startGwBtn.style.display = 'none';
                    this.cancelGwBtn.style.display = 'inline-block';
                } else {
                    alert("No path found between the selected points.");
                }
            }
        });

        this.cancelGwBtn.addEventListener('click', () => {
            this._clearGreenWave();
        });

        this.selectStartBtn.addEventListener('click', () => {
            this.selectingStart = true;
            this.selectingEnd = false;
            this.map.getContainer().style.cursor = 'crosshair';
        });

        this.selectEndBtn.addEventListener('click', () => {
            this.selectingEnd = true;
            this.selectingStart = false;
            this.map.getContainer().style.cursor = 'crosshair';
        });

        this.map.on('click', (e) => {
            if (!this.selectingStart && !this.selectingEnd) return;

            const nearestNode = this._findNearestNode(e.latlng);
            if (!nearestNode) return;

            if (this.selectingStart) {
                this.startNode = nearestNode;
                if (this.startMarker) this.startMarker.remove();
                this.startMarker = L.marker([nearestNode.lat, nearestNode.lon]).addTo(this.map);
                this.startCoordsEl.textContent = `${nearestNode.lat.toFixed(5)}, ${nearestNode.lon.toFixed(5)}`;
                this.selectingStart = false;
            } else if (this.selectingEnd) {
                this.endNode = nearestNode;
                if (this.endMarker) this.endMarker.remove();
                this.endMarker = L.marker([nearestNode.lat, nearestNode.lon]).addTo(this.map);
                this.endCoordsEl.textContent = `${nearestNode.lat.toFixed(5)}, ${nearestNode.lon.toFixed(5)}`;
                this.selectingEnd = false;
            }

            this.map.getContainer().style.cursor = '';
            if (this.startNode && this.endNode) {
                this.startGwBtn.disabled = false;
            }
        });
    }

    _findNearestNode(latlng) {
        let nearestNode = null;
        let minDistance = Infinity;

        for (const nodeId in this.roadNetwork.nodes) {
            const node = this.roadNetwork.nodes[nodeId];
            const distance = latlng.distanceTo(L.latLng(node.lat, node.lon));
            if (distance < minDistance) {
                minDistance = distance;
                nearestNode = { id: nodeId, ...node };
            }
        }
        return nearestNode;
    }

    _h(nodeId1, nodeId2) {
        const node1 = this.roadNetwork.nodes[nodeId1];
        const node2 = this.roadNetwork.nodes[nodeId2];
        return L.latLng(node1.lat, node1.lon).distanceTo(L.latLng(node2.lat, node2.lon));
    }

    _aStarSearch(startId, goalId) {
        let openSet = [startId];
        const cameFrom = {};

        const gScore = {};
        gScore[startId] = 0;

        const fScore = {};
        fScore[startId] = this._h(startId, goalId);

        while (openSet.length > 0) {
            let currentId = openSet.reduce((a, b) => fScore[a] < fScore[b] ? a : b);

            if (currentId === goalId) {
                // Reconstruct path
                const path = [currentId];
                while (cameFrom[currentId]) {
                    currentId = cameFrom[currentId];
                    path.unshift(currentId);
                }
                return path;
            }

            openSet = openSet.filter(id => id !== currentId);

            const currentNode = this.roadNetwork.nodes[currentId];
            const neighborWays = currentNode.ways;

            neighborWays.forEach(wayId => {
                const way = this.roadNetwork.ways[wayId];
                way.nodes.forEach(neighborId => {
                    if (neighborId == currentId) return;

                    const tentative_gScore = gScore[currentId] + this._h(currentId, neighborId);
                    if (!gScore[neighborId] || tentative_gScore < gScore[neighborId]) {
                        cameFrom[neighborId] = currentId;
                        gScore[neighborId] = tentative_gScore;
                        fScore[neighborId] = gScore[neighborId] + this._h(neighborId, goalId);
                        if (!openSet.includes(neighborId)) {
                            openSet.push(neighborId);
                        }
                    }
                });
            });
        }

        return null; // No path found
    }

    _visualizePath(path) {
        this.simulation.hideVehicles();
        const latlngs = path.map(nodeId => {
            const node = this.roadNetwork.nodes[nodeId];
            return [node.lat, node.lon];
        });

        this.pathPolyline = L.polyline(latlngs, { color: 'lime', weight: 6, opacity: 0.8 }).addTo(this.map);
        this.map.fitBounds(this.pathPolyline.getBounds());

        // Add to simulation
        const emergencyVehicle = new EmergencyVehicle(999, this.map, this.roadNetwork, path);
        this.simulation.addVehicle(emergencyVehicle);
        this.emergencyVehicle = emergencyVehicle;
    }

    _clearGreenWave() {
        this.simulation.showVehicles();
        if (this.pathPolyline) this.pathPolyline.remove();
        if (this.emergencyVehicle) this.simulation.removeVehicle(this.emergencyVehicle.id);
        if (this.startMarker) this.startMarker.remove();
        if (this.endMarker) this.endMarker.remove();

        this.path = null;
        this.startNode = null;
        this.endNode = null;
        this.startCoordsEl.textContent = 'Not selected';
        this.endCoordsEl.textContent = 'Not selected';
        this.startGwBtn.disabled = true;
        this.startGwBtn.style.display = 'inline-block';
        this.cancelGwBtn.style.display = 'none';
    }
}

class Vehicle {
    constructor(id, map, roadNetwork) {
        this.id = id;
        this.map = map;
        this.roadNetwork = roadNetwork;
        this.speed = (Math.random() * 30 + 30) / 3600; // km/s

        const startNodeId = this._findRandomStartNode();
        const startNode = this.roadNetwork.nodes[startNodeId];
        this.lat = startNode.lat;
        this.lon = startNode.lon;

        this.marker = L.circleMarker([this.lat, this.lon], { radius: 4, color: 'cyan' }).addTo(map);

        this._planInitialRoute(startNodeId);
    }

    _findRandomStartNode() {
        const nodeIds = Object.keys(this.roadNetwork.nodes);
        return nodeIds[Math.floor(Math.random() * nodeIds.length)];
    }

    _planInitialRoute(startNodeId) {
        const startNode = this.roadNetwork.nodes[startNodeId];
        if (!startNode.ways.length) {
            // This node is not connected, find another one
            const newStartNodeId = this._findRandomStartNode();
            this._planInitialRoute(newStartNodeId);
            return;
        }

        const wayId = startNode.ways[Math.floor(Math.random() * startNode.ways.length)];
        const way = this.roadNetwork.ways[wayId];

        let targetNodeIndex = way.nodes.indexOf(parseInt(startNodeId));
        if (targetNodeIndex === 0) {
            targetNodeIndex = 1;
        } else {
            targetNodeIndex = targetNodeIndex -1;
        }

        const targetNodeId = way.nodes[targetNodeIndex];
        this.route = {
            wayId: wayId,
            targetNodeId: targetNodeId,
            currentNodeIndex: way.nodes.indexOf(parseInt(startNodeId))
        };
    }

    update(deltaTime) {
        if (!this.route) {
            this._planInitialRoute(this._findRandomStartNode());
            if(!this.route) return; // Still no route, skip update
        }

        const targetNode = this.roadNetwork.nodes[this.route.targetNodeId];
        const targetLat = targetNode.lat;
        const targetLon = targetNode.lon;

        const distanceToTarget = L.latLng(this.lat, this.lon).distanceTo(L.latLng(targetLat, targetLon));
        const distanceToMove = this.speed * 1000 * deltaTime;

        if (distanceToMove >= distanceToTarget) {
            this.lat = targetLat;
            this.lon = targetLon;
            this._planInitialRoute(this.route.targetNodeId);
        } else {
            const bearing = calculateBearing(this.lat, this.lon, targetLat, targetLon);
            const newPos = calculateDestinationPoint(this.lat, this.lon, bearing, distanceToMove);
            this.lat = newPos.lat;
            this.lon = newPos.lon;
        }

        this.marker.setLatLng([this.lat, this.lon]);
    }
}

// --- Geodesic Helper Functions ---

function toRadians(degrees) {
    return degrees * Math.PI / 180;
}

function toDegrees(radians) {
    return radians * 180 / Math.PI;
}

function calculateBearing(lat1, lon1, lat2, lon2) {
    const y = Math.sin(toRadians(lon2 - lon1)) * Math.cos(toRadians(lat2));
    const x = Math.cos(toRadians(lat1)) * Math.sin(toRadians(lat2)) -
              Math.sin(toRadians(lat1)) * Math.cos(toRadians(lat2)) * Math.cos(toRadians(lon2 - lon1));
    const brng = Math.atan2(y, x);
    return (toDegrees(brng) + 360) % 360;
}

function calculateDestinationPoint(lat, lon, bearing, distance) {
    const R = 6371e3; // Earth's radius in metres
    const d = distance;

    const lat1 = toRadians(lat);
    const lon1 = toRadians(lon);
    const brng = toRadians(bearing);

    const lat2 = Math.asin(Math.sin(lat1) * Math.cos(d / R) +
                          Math.cos(lat1) * Math.sin(d / R) * Math.cos(brng));
    const lon2 = lon1 + Math.atan2(Math.sin(brng) * Math.sin(d / R) * Math.cos(lat1),
                                 Math.cos(d / R) - Math.sin(lat1) * Math.sin(lat2));

    return { lat: toDegrees(lat2), lon: toDegrees(lon2) };
}


const sampleEvents = [
    {
        name: 'Cricket Match at Kalinga Stadium',
        location: { lat: 20.303, lon: 85.823 },
        time: '2025-09-03 19:00',
        radius: 1500 // meters
    },
    {
        name: 'Concert at Exhibition Ground',
        location: { lat: 20.27, lon: 85.83 },
        time: '2025-09-04 18:00',
        radius: 2000
    },
    {
        name: 'Festival at Lingaraj Temple',
        location: { lat: 20.238, lon: 85.833 },
        time: '2025-09-05 09:00',
        radius: 1000
    }
];

class EventHorizon {
    constructor(map) {
        this.map = map;
        this.events = sampleEvents;
        this.eventListEl = document.getElementById('event-list');

        this._populateEventList();
        this._displayEventsOnMap();
    }

    _populateEventList() {
        this.events.forEach(event => {
            const li = document.createElement('li');
            li.innerHTML = `<strong>${event.name}</strong><br><span>${new Date(event.time).toLocaleString()}</span>`;
            li.addEventListener('click', () => {
                this.map.setView(event.location, 15);
            });
            this.eventListEl.appendChild(li);
        });
    }

    _displayEventsOnMap() {
        this.events.forEach(event => {
            const eventMarker = L.marker(event.location, {
                icon: L.divIcon({
                    className: 'event-icon',
                    html: '<i class="fas fa-calendar-star"></i>',
                    iconSize: [30, 30]
                })
            }).addTo(this.map);

            eventMarker.bindPopup(`<strong>${event.name}</strong><br>${new Date(event.time).toLocaleString()}`);

            L.circle(event.location, {
                radius: event.radius,
                color: 'orange',
                fillColor: 'yellow',
                fillOpacity: 0.2
            }).addTo(this.map);
        });
    }
}

class Simulation {
    constructor(map, roadNetwork) {
        this.map = map;
        this.roadNetwork = roadNetwork;
        this.vehicles = [];
        this.lastTimestamp = 0;
        this.totalSpeed = 0;
        this.co2saved = 0;
    }

    start(numVehicles) {
        for (let i = 0; i < numVehicles; i++) {
            const vehicle = new Vehicle(i, this.map, this.roadNetwork);
            this.vehicles.push(vehicle);
        }
        this.lastTimestamp = performance.now();
        this._loop();
    }

    addVehicle(vehicle) {
        this.vehicles.push(vehicle);
    }

    removeVehicle(vehicleId) {
        const vehicle = this.vehicles.find(v => v.id === vehicleId);
        if (vehicle) {
            vehicle.marker.remove();
            this.vehicles = this.vehicles.filter(v => v.id !== vehicleId);
        }
    }

    hideVehicles() {
        this.vehicles.forEach(v => {
            if (!(v instanceof EmergencyVehicle)) {
                v.marker.remove();
            }
        });
    }

    showVehicles() {
        this.vehicles.forEach(v => {
            if (!(v instanceof EmergencyVehicle)) {
                v.marker.addTo(this.map);
            }
        });
    }

    _loop() {
        const timestamp = performance.now();
        const deltaTime = (timestamp - this.lastTimestamp) / 1000; // in seconds
        this.lastTimestamp = timestamp;

        let currentTotalSpeed = 0;
        this.vehicles.forEach(vehicle => {
            vehicle.update(deltaTime);
            currentTotalSpeed += vehicle.speed * 3600; // km/h
            if (vehicle.speed > 20/3600) {
                this.co2saved += 0.0001 * deltaTime;
            }
        });
        this.totalSpeed = currentTotalSpeed;

        requestAnimationFrame(() => this._loop());
    }
}

class Analytics {
    constructor(simulation) {
        this.simulation = simulation;
        this.avgSpeedEl = document.getElementById('avg-speed');
        this.co2SavedEl = document.getElementById('co2-saved');
        this.activeVehiclesEl = document.getElementById('active-vehicles');

        this.chart = this._initChart();

        setInterval(() => this.update(), 2000);
    }

    _initChart() {
        const ctx = document.getElementById('speed-chart').getContext('2d');
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Average Speed (km/h)',
                    data: [],
                    borderColor: '#4a90e2',
                    backgroundColor: 'rgba(74, 144, 226, 0.2)',
                    fill: true,
                }]
            },
            options: {
                scales: {
                    x: {
                        ticks: { color: '#f0f0f0' }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: { color: '#f0f0f0' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#f0f0f0' }
                    }
                }
            }
        });
    }

    update() {
        const vehicleCount = this.simulation.vehicles.length;
        const avgSpeed = vehicleCount > 0 ? (this.simulation.totalSpeed / vehicleCount) : 0;

        this.avgSpeedEl.textContent = avgSpeed.toFixed(1);
        this.co2SavedEl.textContent = this.simulation.co2saved.toFixed(2);
        this.activeVehiclesEl.textContent = vehicleCount;

        // Update chart
        const now = new Date();
        const timeString = now.toLocaleTimeString();
        this.chart.data.labels.push(timeString);
        this.chart.data.datasets[0].data.push(avgSpeed);

        // Limit chart data to last 30 points
        if (this.chart.data.labels.length > 30) {
            this.chart.data.labels.shift();
            this.chart.data.datasets[0].data.shift();
        }

        this.chart.update();
    }
}

class EmergencyVehicle extends Vehicle {
    constructor(id, map, roadNetwork, path) {
        super(id, map, roadNetwork);
        this.speed = 80 / 3600; // Faster speed: 80 km/h
        this.path = path;
        this.pathIndex = 0;

        const startNode = this.roadNetwork.nodes[this.path[0]];
        this.lat = startNode.lat;
        this.lon = startNode.lon;

        this.marker.remove(); // remove the default marker
        this.marker = L.circleMarker([this.lat, this.lon], { radius: 6, color: 'red', fillColor: 'red', fillOpacity: 1 }).addTo(map);
    }

    update(deltaTime) {
        if (this.pathIndex >= this.path.length - 1) {
            // Reached destination
            return;
        }

        const targetNodeId = this.path[this.pathIndex + 1];
        const targetNode = this.roadNetwork.nodes[targetNodeId];
        const targetLat = targetNode.lat;
        const targetLon = targetNode.lon;

        const distanceToTarget = L.latLng(this.lat, this.lon).distanceTo(L.latLng(targetLat, targetLon));
        const distanceToMove = this.speed * 1000 * deltaTime;

        if (distanceToMove >= distanceToTarget) {
            this.lat = targetLat;
            this.lon = targetLon;
            this.pathIndex++;
        } else {
            const bearing = calculateBearing(this.lat, this.lon, targetLat, targetLon);
            const newPos = calculateDestinationPoint(this.lat, this.lon, bearing, distanceToMove);
            this.lat = newPos.lat;
            this.lon = newPos.lon;
        }

        this.marker.setLatLng([this.lat, this.lon]);
    }
}

class Weather {
    constructor() {
        this.container = document.getElementById('weather-panel-content');
        this.fetchWeather();
    }

    fetchWeather() {
        // This function is designed to use mock data by default.
        // To use a real API, uncomment the following lines and insert your key.
        /*
        const apiKey = 'YOUR_API_KEY';
        const lat = 20.2961;
        const lon = 85.8245;
        const url = `https://api.weatherapi.com/v1/current.json?key=${apiKey}&q=${lat},${lon}`;

        fetch(url)
            .then(response => {
                if (!response.ok) throw new Error('Weather data not available');
                return response.json();
            })
            .then(data => this._displayRealWeather(data))
            .catch(error => {
                console.error('Error fetching weather:', error);
                this._displayMockWeather();
            });
        */
        // Default to mock data
        this._displayMockWeather();
    }

    _displayRealWeather(data) {
        const weather = data.current;
        this.container.innerHTML = `
            <h1>Weather in Bhubaneswar</h1>
            <p><strong>Temperature:</strong> ${weather.temp_c}°C</p>
            <p><strong>Condition:</strong> ${weather.condition.text}</p>
            <img src="https:${weather.condition.icon}" alt="${weather.condition.text}">
        `;
    }

    _displayMockWeather() {
        this.container.innerHTML = `
            <h1>Weather in Bhubaneswar</h1>
            <p><strong>Temperature:</strong> 28°C</p>
            <p><strong>Condition:</strong> Clear Sky</p>
            <p><em>(Mock Data)</em></p>
            <img src="https://cdn.weatherapi.com/weather/64x64/day/113.png" alt="Clear Sky">
        `;
    }
}
