document.addEventListener('DOMContentLoaded', function () {
  if (typeof mapboxgl === 'undefined') {
    console.error('Mapbox GL JS not loaded.');
    const mapDiv = document.getElementById('map');
    if (mapDiv) mapDiv.innerHTML = '<p style="text-align: center; padding: 20px;">Map resources failed to load. Please check your internet connection or ad-blocker settings and try again.</p>';
    alert('Could not load map resources.');
    return;
  }

  // Helper function to create a circle polygon from a point, used for the extrusion base
  function createCirclePolygon(center, radiusInMeters, points = 32) {
    const coords = {
        latitude: center[1],
        longitude: center[0]
    };

    const km = radiusInMeters / 1000;
    const ret = [];
    const distanceX = km / (111.320 * Math.cos(coords.latitude * Math.PI / 180));
    const distanceY = km / 111.132;

    let theta, x, y;
    for (let i = 0; i < points; i++) {
        theta = (i / points) * (2 * Math.PI);
        x = distanceX * Math.cos(theta);
        y = distanceY * Math.sin(theta);
        ret.push([coords.longitude + x, coords.latitude + y]);
    }
    ret.push(ret[0]);

    return [ret]; // GeoJSON Polygon coordinates format
  }

  let map;
  let mapboxApiKey = localStorage.getItem('mapboxApiKey');

  const apiKeyInput = document.getElementById('apiKey');
  const setApiKeyButton = document.getElementById('setApiKey');
  const dateInput = document.getElementById('date');
  const minMagInput = document.getElementById('minMag');
  const minMagValueSpan = document.getElementById('minMagValue');
  const filterButton = document.getElementById('filter');
  const statusText = document.getElementById('status-text');
  const propertiesDisplay = document.getElementById('properties-display');
  const mapPlaceholder = document.querySelector('.map-placeholder');
  const earthquakeTableBody = document.querySelector('#earthquake-table tbody');

  function updateStatusPanel(status, properties = null) {
    statusText.textContent = status;
    if (properties) {
      propertiesDisplay.textContent = JSON.stringify(properties, null, 2);
    }
  }

  function setTableMessage(message) {
    if (!earthquakeTableBody) return;
    const colSpan = 2;
    earthquakeTableBody.innerHTML = `<tr><td colspan="${colSpan}" style="text-align: center; padding: 20px;">${message}</td></tr>`;
  }

  function setControlsDisabled(disabled) {
    dateInput.disabled = disabled;
    minMagInput.disabled = disabled;
    filterButton.disabled = disabled;
  }

  // Initially disable controls. They will be enabled once the map is successfully loaded.
  // The 'disabled' attribute is already set in the HTML, this is for dynamic state changes.
  setControlsDisabled(true);

  if (mapboxApiKey) {
    apiKeyInput.value = mapboxApiKey;
    initializeMap(mapboxApiKey);
  } else {
    setTableMessage('Enter API Key to load data.');
    updateStatusPanel('Waiting for API Key.', { status: 'Not initialized' });
  }

  // Set default date to today
  const today = new Date();
  const yyyy = today.getFullYear();
  const mm = String(today.getMonth() + 1).padStart(2, '0'); // Months are 0-based
  const dd = String(today.getDate()).padStart(2, '0');
  dateInput.value = `${yyyy}-${mm}-${dd}`;

  minMagInput.addEventListener('input', () => {
    minMagValueSpan.textContent = parseFloat(minMagInput.value).toFixed(1);
  });

  setApiKeyButton.addEventListener('click', () => {
    const key = apiKeyInput.value.trim();
    if (!key) {
      alert('Please enter a valid MapBox API Key.');
      return;
    }

    localStorage.setItem('mapboxApiKey', key);
    mapboxApiKey = key;

    // To ensure the new key is applied correctly and to handle all cases
    // (e.g., updating a valid key, or replacing an invalid one), the simplest
    // and most robust approach is to remove the old map instance and create a new one.
    if (map) map.remove();
    initializeMap(key);
  });

  filterButton.addEventListener('click', loadEarthquakeData);

  function updateMapProperties() {
    if (!map) return;
    const center = map.getCenter();
    const properties = {
      status: 'Idle',
      zoom: map.getZoom().toFixed(2),
      center: {
        lng: center.lng.toFixed(4),
        lat: center.lat.toFixed(4)
      },
      bounds: map.getBounds().toArray().map(c => c.map(n => n.toFixed(4)))
    };
    // We only update the properties, not the main status text, which might be showing "Data loaded"
    propertiesDisplay.textContent = JSON.stringify(properties, null, 2);
  }

  function initializeMap(apiKey) {
    updateStatusPanel('Initializing map...', { key_provided: !!apiKey });
    mapboxgl.accessToken = apiKey;

    // Hide the placeholder before initializing the map
    if (mapPlaceholder) mapPlaceholder.style.display = 'none';

    map = new mapboxgl.Map({
      container: 'map',
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [-98.5795, 39.8283], // Center of the US
      zoom: 3,
      pitch: 45, // Set a pitch for 3D viewing
      bearing: -17.6, // Rotate the map for a better perspective
      // Make sure the map container is ready
      transformRequest: (url, resourceType) => ({ url, resourceType })
    });

    map.on('load', () => {
      // Add a light source to make the 3D look better
      map.setLight({anchor: 'viewport', color: 'white', intensity: 0.4});

      // Add a source for the earthquake data.
      map.addSource('earthquakes', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: []
        }
      });

      // Add a layer to display the earthquakes as 3D cylinders.
      map.addLayer({
        id: 'earthquakes-layer',
        type: 'fill-extrusion',
        source: 'earthquakes',
        paint: {
          'fill-extrusion-color': [
            'interpolate',
            ['linear'],
            ['get', 'mag'],
            1, '#ff8040', // Lightest Red
            3, '#ff4020', // Light Red
            5, '#ff0000', // Medium Red
            7, '#ff4020', // Dark Red
            9, '#ff8040'  // Darkest Red
          ],
          'fill-extrusion-height': [
            'interpolate',
            ['linear'],
            ['get', 'mag'],
            1, 100000,   // mag 1 = 100km high
            8, 5000000  // mag 8 = 5000km high
          ],
          'fill-extrusion-base': 0,
          'fill-extrusion-opacity': 0.85
        }
      });

      // Create a popup, but don't add it to the map yet.
      const popup = new mapboxgl.Popup({
        closeButton: false,
        closeOnClick: false
      });

      map.on('mouseenter', 'earthquakes-layer', (e) => {
        // Change the cursor style as a UI indicator.
        map.getCanvas().style.cursor = 'pointer';

        const { mag, place, time } = e.features[0].properties;

        popup.setLngLat(e.lngLat)
          .setHTML(`<strong>Magnitude: ${mag}</strong><br>Location: ${place}<br>Time: ${new Date(time).toLocaleString()}`)
          .addTo(map);
      });

      map.on('mouseleave', 'earthquakes-layer', () => {
        map.getCanvas().style.cursor = '';
        popup.remove();
      });

      // Add event listeners to update properties on map interaction
      map.on('moveend', () => updateMapProperties());
      map.on('zoomend', () => updateMapProperties());

      // Enable controls now that the map is loaded
      setControlsDisabled(false);

      // The 'load' event fires when the map's style is ready. However, sources
      // added within the 'load' event may not be ready for manipulation immediately.
      // We use map.once('idle', ...) to ensure the map is fully stable and all
      // sources are loaded before we fetch and display the initial data.
      map.once('idle', () => {
        updateStatusPanel('Map idle. Loading initial earthquake data...');
        loadEarthquakeData();
      });
    });

    map.on('error', (e) => {
        if (e.error && e.error.message.includes('accessToken')) {
            alert('Invalid MapBox API Key. Please provide a valid key.');
            updateStatusPanel('Error: Invalid API Key.', {
              error: e.error.message,
              status: 'Failed to load'
            });
            if(map) map.remove();
            map = null;
            localStorage.removeItem('mapboxApiKey');
            apiKeyInput.value = '';
            setControlsDisabled(true);
            // Show the placeholder again
            if (mapPlaceholder) mapPlaceholder.style.display = 'flex';
            setTableMessage('Map failed to load.');
        }
    });
  }

  async function loadEarthquakeData() {
    if (!map) {
      alert('Please set your MapBox API key first.');
      return;
    }

    const selectedDate = dateInput.value;
    if (!selectedDate) {
      alert('Please select a date.');
      return;
    }

    const minMagnitude = minMagInput.value;
    updateStatusPanel(`Fetching earthquakes since ${selectedDate}...`, {
      minMagnitude: minMagnitude,
      status: 'Loading data'
    });

    // Clear table and show loading state
    setTableMessage('Loading...');

    const startTime = new Date(selectedDate);
    startTime.setUTCHours(0, 0, 0, 0);
    
    const endTime = new Date(startTime);
    endTime.setDate(startTime.getDate() + 1);

    const usgsApiUrl = `https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=${startTime.toISOString()}&endtime=${endTime.toISOString()}&minmagnitude=${minMagnitude}`;

    try {
      const response = await fetch(usgsApiUrl);
      const geojsonData = await response.json();

      // Update table with original point data before transforming for map
      updateEarthquakeTable(geojsonData.features);

      // Transform points to polygons for extrusion
      const extrudedFeatures = geojsonData.features.map(feature => {
          if (feature.geometry.type === 'Point') {
              const coordinates = feature.geometry.coordinates;
              const mag = feature.properties.mag;
              // Define radius for the base of the cylinder based on magnitude
              const radius = mag * mag * 10000; // radius in meters (mag^2 * 10km)
              const polygonCoords = createCirclePolygon(coordinates, radius);
              return {
                  ...feature,
                  geometry: {
                      type: 'Polygon',
                      coordinates: polygonCoords
                  }
              };
          }
          return feature;
      });

      const earthquakeSource = map.getSource('earthquakes');
      if (earthquakeSource) {
        earthquakeSource.setData({ type: 'FeatureCollection', features: extrudedFeatures });
        const message = `Loaded ${geojsonData.features.length} earthquakes.`;
        updateStatusPanel(message);
        updateMapProperties();
      }
    } catch (error) {
      console.error('Error fetching earthquake data:', error);
      alert('Failed to load earthquake data. See console for details.');
      updateStatusPanel('Failed to load earthquake data.', {
        error: error.message
      });
      setTableMessage('Failed to load data.');
    }
  }

  function updateEarthquakeTable(features) {
    if (!earthquakeTableBody) return;
    earthquakeTableBody.innerHTML = ''; // Clear existing rows

    if (!features || features.length === 0) {
      setTableMessage('No earthquakes found for this date.');
      return;
    }

    const top10 = features
      .slice() // Create a copy to avoid modifying the original array
      .sort((a, b) => b.properties.mag - a.properties.mag)
      .slice(0, 10);

    top10.forEach(feature => {
      const { mag, place } = feature.properties;
      const row = earthquakeTableBody.insertRow();
      const magCell = row.insertCell();
      const placeCell = row.insertCell();

      magCell.textContent = mag.toFixed(1);
      magCell.style.fontWeight = 'bold';
      placeCell.textContent = place;
    });
  }
});