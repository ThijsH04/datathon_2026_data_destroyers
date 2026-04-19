/**
 * Strado Scoring Engine
 *
 * Computes livability and activity composite scores from per-category POI counts.
 * Three swappable algorithms for livability; activity always uses density.
 */

// Categories NOT included in either composite score:
// - playgrounds: subset of parks, would double-count
// - cycling: infrastructure metric, not a neighborhood amenity
// - car_infra: anti-correlated with walkability goals
// - pet_friendly: niche, would dilute the composite
// These are available as standalone category views on the map.

// Category classifications
const ESSENTIAL_CATEGORIES = [
  'grocery', 'healthcare', 'transit', 'parks', 'education',
  'safety', 'early_education', 'personal_care', 'financial'
];

const LIFESTYLE_CATEGORIES = [
  'nightlife', 'cafes', 'dining', 'culture', 'shopping',
  'sports', 'beaches', 'accommodation', 'coworking'
];

// Thresholds: how many POIs of each type constitutes "good coverage"
// These are tuned for H3 res-9 cells (~174m edge) with k=1 expansion
const ESSENTIAL_THRESHOLDS = {
  grocery:        { min: 1, good: 3 },
  healthcare:     { min: 1, good: 2 },
  transit:        { min: 1, good: 3 },
  parks:          { min: 1, good: 2 },
  education:      { min: 1, good: 2 },
  safety:         { min: 1, good: 1 },
  early_education:{ min: 1, good: 2 },
  personal_care:  { min: 1, good: 2 },
  financial:      { min: 1, good: 2 },
};

/**
 * Score input: object with category keys and integer count values.
 * Example: { grocery: 3, healthcare: 1, transit: 5, parks: 2, nightlife: 12 }
 *
 * Score output: { score: 0-100, grade: 'A+'..'F', missing: ['safety'], present: ['grocery', ...] }
 */

function scoreToGrade(score) {
  if (score >= 95) return 'A+';
  if (score >= 85) return 'A';
  if (score >= 75) return 'B+';
  if (score >= 65) return 'B';
  if (score >= 55) return 'C+';
  if (score >= 45) return 'C';
  if (score >= 30) return 'D';
  return 'F';
}

// ============================================================
// Algorithm 1: Binary Checklist
// Each essential category is "present" (count >= 1) or "missing".
// Score = percentage of categories present.
// ============================================================
function scoreLivabilityBinary(counts) {
  const missing = [];
  const present = [];
  for (const cat of ESSENTIAL_CATEGORIES) {
    const count = counts[cat] || 0;
    if (count >= 1) {
      present.push(cat);
    } else {
      missing.push(cat);
    }
  }
  const score = Math.round((present.length / ESSENTIAL_CATEGORIES.length) * 100);
  return { score, grade: scoreToGrade(score), missing, present, algorithm: 'binary' };
}

// ============================================================
// Algorithm 2: Tiered Thresholds
// Each category scores 0 (missing), 0.6 (min threshold), or 1.0 (good threshold).
// Score = weighted percentage.
// ============================================================
function scoreLivabilityTiered(counts) {
  const missing = [];
  const present = [];
  let totalPoints = 0;
  for (const cat of ESSENTIAL_CATEGORIES) {
    const count = counts[cat] || 0;
    const thresh = ESSENTIAL_THRESHOLDS[cat];
    if (count >= thresh.good) {
      totalPoints += 1.0;
      present.push(cat);
    } else if (count >= thresh.min) {
      totalPoints += 0.6;
      present.push(cat);
    } else {
      missing.push(cat);
    }
  }
  const score = Math.round((totalPoints / ESSENTIAL_CATEGORIES.length) * 100);
  return { score, grade: scoreToGrade(score), missing, present, algorithm: 'tiered' };
}

// ============================================================
// Algorithm 3: Diminishing Returns
// Each POI in an essential category adds score with logarithmic falloff.
// First POI adds ~15 points, second adds ~9, fifth adds ~4.
// ============================================================
function scoreLivabilityDiminishing(counts) {
  const missing = [];
  const present = [];
  let totalScore = 0;
  const maxPerCategory = 100 / ESSENTIAL_CATEGORIES.length; // ~11.1 per category

  for (const cat of ESSENTIAL_CATEGORIES) {
    const count = counts[cat] || 0;
    if (count === 0) {
      missing.push(cat);
    } else {
      present.push(cat);
      // log2(count + 1) / log2(threshold.good + 1) * maxPerCategory
      const thresh = ESSENTIAL_THRESHOLDS[cat];
      const normalized = Math.min(1.0, Math.log2(count + 1) / Math.log2(thresh.good + 2));
      totalScore += normalized * maxPerCategory;
    }
  }
  const score = Math.round(Math.min(100, totalScore));
  return { score, grade: scoreToGrade(score), missing, present, algorithm: 'diminishing' };
}

// ============================================================
// Activity Score (density-based, single algorithm)
// Sum of log-scaled counts across lifestyle categories, normalized to 0-100.
// ============================================================
function scoreActivity(counts) {
  let totalRaw = 0;
  const breakdown = {};
  for (const cat of LIFESTYLE_CATEGORIES) {
    const count = counts[cat] || 0;
    totalRaw += count;
    breakdown[cat] = count;
  }
  // Log scale: score = 15 * log2(total + 1), capped at 100
  const score = Math.round(Math.min(100, 15 * Math.log2(totalRaw + 1)));
  return { score, grade: scoreToGrade(score), breakdown, algorithm: 'density' };
}

// ============================================================
// Main entry point: compute both composites
// ============================================================
const LIVABILITY_ALGORITHMS = {
  binary: scoreLivabilityBinary,
  tiered: scoreLivabilityTiered,
  diminishing: scoreLivabilityDiminishing,
};

function computeScores(counts, livabilityAlgorithm) {
  counts = counts || {};
  // Clamp negative values to 0
  for (const key of Object.keys(counts)) {
    if (typeof counts[key] === 'number' && counts[key] < 0) counts[key] = 0;
  }
  const algo = livabilityAlgorithm || 'tiered';
  const livFn = LIVABILITY_ALGORITHMS[algo] || scoreLivabilityTiered;
  return {
    livability: livFn(counts),
    activity: scoreActivity(counts),
  };
}

// ============================================================
// Hex color expression for composite scores
// Returns a MapLibre step expression that colors hexes by score.
// ============================================================
function compositeColorExpr(scoreProperty) {
  return [
    'step', ['coalesce', ['get', scoreProperty], 0],
    '#374151',   // 0: no data (dark gray)
    1, '#a50026',  // F (1-29)
    30, '#d73027', // D (30-44)
    45, '#f46d43', // C (45-54)
    55, '#fdae61', // C+ (55-64)
    65, '#fee08b', // B (65-74)
    75, '#a6d96a', // B+ (75-84)
    85, '#1a9850', // A (85-94)
    95, '#006837', // A+ (95-100)
  ];
}

// Category type check
function isEssentialCategory(category) {
  return ESSENTIAL_CATEGORIES.includes(category);
}

function isLifestyleCategory(category) {
  return LIFESTYLE_CATEGORIES.includes(category);
}

// ============================================================
// Completeness color expression for essential categories
// Green = present (count >= 1), gray = missing
// ============================================================
function completenessColorExpr(categoryProperty) {
  return [
    'step', ['coalesce', ['get', categoryProperty], 0],
    '#374151',      // 0: missing (dark gray)
    1, '#1a9850',   // 1+: present (green)
  ];
}

// ============================================================
// Density color expression for lifestyle categories (red gradient)
// ============================================================
function densityColorExpr(categoryProperty, sensitivity) {
  var s = sensitivity || 10;
  var t1 = 1;
  var t2 = Math.max(t1 + 1, Math.round(s * 0.3));
  var t3 = Math.max(t2 + 1, Math.round(s * 0.6));
  var t4 = Math.max(t3 + 1, Math.round(s));
  var t5 = Math.max(t4 + 1, Math.round(s * 2));
  return [
    'step', ['coalesce', ['get', categoryProperty], 0],
    'transparent',
    t1, '#fee5d9',
    t2, '#fcae91',
    t3, '#fb6a4a',
    t4, '#de2d26',
    t5, '#a50f15',
  ];
}

document.addEventListener('DOMContentLoaded', function() {
// ============================================================
// CONSTANTS
// ============================================================
const API_BASE = window.location.origin;
const MIN_ZOOM = 9;
const HEX_CACHE_MAX = 200;

const CATEGORIES = {
  grocery:         'Grocery',
  dining:          'Dining',
  cafes:           'Cafes',
  nightlife:       'Nightlife',
  healthcare:      'Healthcare',
  early_education: 'Early Education',
  education:       'Schools',
  parks:           'Parks',
  playgrounds:     'Playgrounds',
  sports:          'Sports',
  cycling:         'Cycling',
  transit:         'Transit',
  car_infra:       'Parking',
  culture:         'Culture',
  pet_friendly:    'Pet Friendly',
  financial:       'Financial',
  safety:          'Safety',
  shopping:        'Shopping',
  personal_care:   'Personal Care',
  accommodation:   'Accommodation',
  coworking:       'Coworking',
  beaches:         'Beaches',
};

const CATEGORY_COLORS = {
  grocery: '#e41a1c', dining: '#377eb8', cafes: '#984ea3', nightlife: '#4daf4a',
  healthcare: '#ff7f00', early_education: '#ffff33', education: '#a65628',
  parks: '#1a9850', playgrounds: '#f781bf', sports: '#999999',
  cycling: '#1D9E75', transit: '#66c2a5', car_infra: '#8da0cb',
  culture: '#e78ac3', pet_friendly: '#a6d854', financial: '#ffd92f',
  safety: '#fc8d62', shopping: '#b3b3b3', personal_care: '#bebada',
  accommodation: '#fb8072', coworking: '#80b1d3', beaches: '#ffd92f',
};

const OTHER_CATEGORIES = ['playgrounds', 'cycling', 'car_infra', 'pet_friendly'];

// ESSENTIAL_CATEGORIES, LIFESTYLE_CATEGORIES, ESSENTIAL_THRESHOLDS,
// scoreToGrade, scoreLivabilityTiered, scoreActivity, computeScores,
// isEssentialCategory, isLifestyleCategory -- all provided by /static/js/scoring.js

// ============================================================
// COLOR EXPRESSIONS
// ============================================================
function compositeColorExpr(scoreProperty) {
  return [
    'step', ['coalesce', ['get', scoreProperty], 0],
    '#374151',
    1, '#a50026',
    30, '#d73027',
    45, '#f46d43',
    55, '#fdae61',
    65, '#fee08b',
    75, '#a6d96a',
    85, '#1a9850',
    95, '#006837',
  ];
}

function completenessColorExpr(categoryProperty) {
  return [
    'step', ['coalesce', ['get', categoryProperty], 0],
    '#374151',
    1, '#1a9850',
  ];
}

function densityColorExpr(categoryProperty) {
  return [
    'step', ['coalesce', ['get', categoryProperty], 0],
    'transparent',
    1, '#fee5d9',
    3, '#fcae91',
    6, '#fb6a4a',
    10, '#de2d26',
    20, '#a50f15',
  ];
}

function gradeColor(grade) {
  const gradeColors = {
    'A+': '#006837', 'A': '#1a9850',
    'B+': '#66bd63', 'B': '#a6d96a',
    'C+': '#fdae61', 'C': '#f46d43',
    'D': '#d73027', 'F': '#a50026',
  };
  return gradeColors[grade] || '#374151';
}

// ============================================================
// BASEMAPS
// ============================================================
const BASEMAPS = {
  light: { name: 'Light', style: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json' },
  dark: { name: 'Dark', style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json' },
  satellite: { name: 'Satellite', style: { version: 8, sources: { sat: { type: 'raster', tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'], tileSize: 256, attribution: 'Esri' }}, layers: [{ id: 'satellite', type: 'raster', source: 'sat' }] } },
};

// ============================================================
// SETTINGS (simplified)
// ============================================================
const savedSettings = JSON.parse(localStorage.getItem('strado_map') || '{}');
let currentBasemap = (savedSettings.basemap && BASEMAPS[savedSettings.basemap]) ? savedSettings.basemap : 'light';
let hexOpacity = 0.22;
let currentScoreMode = savedSettings.mode || 'livability';
let activeCategory = null;
let hexRes = 9;
let hexK = 1;

function saveSettings() {
  localStorage.setItem('strado_map', JSON.stringify({
    basemap: currentBasemap,
    mode: currentScoreMode,
  }));
}

// ============================================================
// PMTiles protocol
const pmtilesProtocol = new pmtiles.Protocol();
maplibregl.addProtocol("pmtiles", pmtilesProtocol.tile);

// Tile source URLs
const TILES_BASE = 'pmtiles://https://tiles.strado.info';

// MAP INIT
// ============================================================
const map = new maplibregl.Map({
  container: 'map',
  style: BASEMAPS[currentBasemap].style,
  center: [12.4964, 41.9028],
  zoom: 13,
  attributionControl: false,
});
map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-right');
map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');

// ============================================================
// UTILITIES
// ============================================================
function escapeHtml(str) {
  if (!str) return '';
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

// ============================================================
// HEX TILE STATE
// ============================================================
const hexTileCache = new Map();
let hexDebounceTimer = null;
const _hexControllers = new Map();
let _poiController = null;

function hexCacheSet(key, value) {
  if (hexTileCache.has(key)) {
    hexTileCache.delete(key);
  } else if (hexTileCache.size >= HEX_CACHE_MAX) {
    const firstKey = hexTileCache.keys().next().value;
    hexTileCache.delete(firstKey);
  }
  hexTileCache.set(key, value);
}

// ============================================================
// HEX TILE GEOMETRY
// ============================================================
function getVisibleTiles(map, z, extraTiles = 1) {
  // extraTiles: prefetch margin -- 1 means fetch 1 extra tile in each direction
  const bounds = map.getBounds();
  const n = Math.pow(2, z);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const xMin = clamp(Math.floor((bounds.getWest() + 180) / 360 * n) - extraTiles, 0, n - 1);
  const xMax = clamp(Math.floor((bounds.getEast() + 180) / 360 * n) + extraTiles, 0, n - 1);
  const latToY = lat => {
    const rad = lat * Math.PI / 180;
    return Math.floor((1 - Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / Math.PI) / 2 * n);
  };
  const yMin = clamp(latToY(bounds.getNorth()) - extraTiles, 0, n - 1);
  const yMax = clamp(latToY(bounds.getSouth()) + extraTiles, 0, n - 1);
  const tiles = [];
  for (let x = xMin; x <= xMax; x++) {
    for (let y = yMin; y <= yMax; y++) {
      tiles.push({ z, x, y });
    }
  }
  return tiles;
}

function hexTileZoom() {
  const z = map.getZoom();
  if (z < MIN_ZOOM) return null;
  hexRes = 9;
  // Adaptive tile zoom: keep tile count reasonable (~4-25 tiles per view).
  // Lower map zoom -> lower tile zoom (bigger tiles, fewer requests).
  if (z >= 13) return 13;
  if (z >= 12) return 12;
  if (z >= 11) return 11;
  return 10;
}

// ============================================================
// HEX SOURCE / LAYER
// ============================================================
function addHexSource() {
  if (map.getSource('hex-counts')) return;
  // PMTiles vector tile source (pre-computed hex cells with scores)
  map.addSource('hex-counts', {
    type: 'vector',
    url: TILES_BASE + '/hex-counts.pmtiles',
  });
  map.addLayer({
    id: 'hex-fill',
    type: 'fill',
    source: 'hex-counts',
    'source-layer': 'hex',
    layout: { visibility: 'visible' },
    paint: {
      'fill-color': compositeColorExpr('_livability'),
      'fill-opacity': hexOpacity,
    },
  });
  map.addLayer({
    id: 'hex-outline',
    type: 'line',
    source: 'hex-counts',
    'source-layer': 'hex',
    layout: { visibility: 'visible' },
    paint: {
      'line-color': 'rgba(255,255,255,0.06)',
      'line-width': 0.5,
    },
  });

  // POI vector tile source (shown when category active)
  map.addSource('pois-tiles', {
    type: 'vector',
    url: TILES_BASE + '/pois.pmtiles',
  });
  map.addLayer({
    id: 'poi-dots',
    type: 'circle',
    source: 'pois-tiles',
    'source-layer': 'pois',
    layout: { visibility: 'none' },
    minzoom: 10,
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['zoom'], 10, 2.5, 14, 5, 16, 7],
      'circle-color': '#1d9e75',
      'circle-opacity': 0.8,
      'circle-stroke-width': 0.5,
      'circle-stroke-color': 'rgba(0,0,0,0.3)',
    },
  });

  map.on('click', 'hex-fill', onHexClick);
  map.on('mouseenter', 'hex-fill', () => { map.getCanvas().style.cursor = 'pointer'; });
  map.on('mouseleave', 'hex-fill', () => { map.getCanvas().style.cursor = ''; });

  // POI popup on click
  map.on('click', 'poi-dots', (e) => {
    if (!e.features || !e.features.length) return;
    const p = e.features[0].properties;
    new maplibregl.Popup({ offset: 10, className: 'poi-popup' })
      .setLngLat(e.lngLat)
      .setHTML('<b>' + escapeHtml(p.name || 'Unnamed') + '</b><br>' + escapeHtml(p.type || ''))
      .addTo(map);
  });
  map.on('mouseenter', 'poi-dots', () => { map.getCanvas().style.cursor = 'pointer'; });
  map.on('mouseleave', 'poi-dots', () => { map.getCanvas().style.cursor = ''; });
}

function setHexVisibility(visible) {
  if (!map.getLayer('hex-fill')) return;
  const v = visible ? 'visible' : 'none';
  map.setLayoutProperty('hex-fill', 'visibility', v);
  map.setLayoutProperty('hex-outline', 'visibility', v);
}

function updateHexLayerColor(colorExpr) {
  if (!map.getLayer('hex-fill')) return;
  map.setPaintProperty('hex-fill', 'fill-color', colorExpr);
}

function updateHexLayerOpacity(opacity) {
  if (!map.getLayer('hex-fill')) return;
  map.setPaintProperty('hex-fill', 'fill-opacity', opacity);
}

// ============================================================
// HEX TILE LOADING
// ============================================================
// PMTiles: MapLibre handles tile loading natively. No fetch needed.
function loadHexTiles() {
  // Just manage zoom hint visibility -- MapLibre loads vector tiles automatically
  const z = map.getZoom();
  if (z < MIN_ZOOM) {
    setHexVisibility(false);
    document.getElementById('zoom-hint').classList.add('visible');
  } else {
    setHexVisibility(true);
    document.getElementById('zoom-hint').classList.remove('visible');
  }
}

// PMTiles: no render needed -- MapLibre renders vector tiles directly.
// Scores are pre-computed in the tile properties.
function renderHexFeatures() { /* no-op */ }

function scheduleHexLoad() {
  clearTimeout(hexDebounceTimer);
  hexDebounceTimer = setTimeout(loadHexTiles, 150);
}

// Keep URL in sync with map state so copy-paste always works
let _urlUpdateTimer = null;
function updateUrlState() {
  clearTimeout(_urlUpdateTimer);
  _urlUpdateTimer = setTimeout(() => {
    const c = map.getCenter();
    const z = Math.round(map.getZoom());
    const params = new URLSearchParams();
    params.set('lat', c.lat.toFixed(4));
    params.set('lon', c.lng.toFixed(4));
    params.set('zoom', z);
    if (typeof activeCategory !== 'undefined' && activeCategory) params.set('category', activeCategory);
    if (typeof currentScoreMode !== 'undefined' && currentScoreMode !== 'livability') params.set('mode', currentScoreMode);
    if (typeof currentBasemap !== 'undefined' && currentBasemap !== 'light') params.set('basemap', currentBasemap);
    history.replaceState(null, '', `/map?${params.toString()}`);
  }, 500);
}

// ============================================================
// POI LOADING (for selected category)
// ============================================================
let _poiListeners = [];

function clearPoiListeners() {
  _poiListeners.forEach(({ event, layer, fn }) => {
    try { map.off(event, layer, fn); } catch(e) {}
  });
  _poiListeners = [];
}

function removePoisLayer() {
  clearPoiListeners();
  if (map.getLayer('pois-active-layer')) map.removeLayer('pois-active-layer');
  if (map.getSource('pois-active')) map.removeSource('pois-active');
}

function getBbox() {
  const b = map.getBounds();
  return `${b.getWest()},${b.getSouth()},${b.getEast()},${b.getNorth()}`;
}

function loadPois(category) {
  // PMTiles: filter the pre-loaded vector tile layer instead of fetching from API
  if (!map.getLayer('poi-dots')) return;
  const color = CATEGORY_COLORS[category] || '#1d9e75';
  map.setFilter('poi-dots', ['==', ['get', 'cat'], category]);
  map.setPaintProperty('poi-dots', 'circle-color', color);
  map.setLayoutProperty('poi-dots', 'visibility', 'visible');
}

function removePoisLayer() {
  if (map.getLayer('poi-dots')) {
    map.setLayoutProperty('poi-dots', 'visibility', 'none');
    map.setFilter('poi-dots', null);
  }
}

// ============================================================
// HEX CLICK -> SCORE CARD
// ============================================================
let _selectedMarker = null;

function onHexClick(e) {
  if (!e.features || !e.features.length) return;
  const props = e.features[0].properties;
  const coords = e.lngLat;

  // Add/move marker using Strado favicon SVG (map pin)
  if (_selectedMarker) _selectedMarker.remove();
  const markerEl = document.createElement('div');
  markerEl.style.cssText = 'width:32px;height:32px;cursor:pointer;filter:drop-shadow(0 2px 6px rgba(0,0,0,0.4));';
  markerEl.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32"><path d="M16 2C10.5 2 6 6.5 6 12c0 3.5 2.2 7.5 5.3 11.5 1.4 1.8 2.8 3.3 3.8 4.3.3.3.6.5.9.7.3-.2.6-.4.9-.7 1-1 2.4-2.5 3.8-4.3C23.8 19.5 26 15.5 26 12 26 6.5 21.5 2 16 2z" fill="#0F6E56"/><circle cx="16" cy="11.5" r="5.5" fill="#1D9E75" opacity="0.35"/><circle cx="16" cy="11.5" r="2.8" fill="#E1F5EE"/></svg>`;
  _selectedMarker = new maplibregl.Marker({ element: markerEl, anchor: 'bottom' })
    .setLngLat([coords.lng, coords.lat])
    .addTo(map);

  // Store coords for share link
  window._selectedLat = coords.lat.toFixed(5);
  window._selectedLon = coords.lng.toFixed(5);
  window._selectedZoom = Math.round(map.getZoom());

  showScoreCard(props);
  reverseGeocodeAddress(coords.lat, coords.lng);
  if (typeof umami !== 'undefined') umami.track('hex-click', {lat: coords.lat.toFixed(3), lon: coords.lng.toFixed(3)});
}

function reverseGeocodeAddress(lat, lon) {
  const el = document.getElementById('score-address');
  el.innerHTML = '<span style="color:var(--text-dim);opacity:0.5;">Loading address...</span>';
  fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&zoom=18&addressdetails=1`, {
    headers: { 'Accept-Language': 'en' }
  })
    .then(r => r.json())
    .then(data => {
      const a = data.address || {};
      const parts = [a.road, a.house_number].filter(Boolean);
      const area = a.suburb || a.neighbourhood || a.quarter || a.city_district || '';
      const city = a.city || a.town || a.village || '';
      let address = parts.join(' ');
      if (area) address += (address ? ', ' : '') + area;
      if (city && city !== area) address += (address ? ', ' : '') + city;
      el.innerHTML = address ? `&#128205; ${address}` : '';
    })
    .catch(() => { el.innerHTML = ''; });
}

function showScoreCard(props) {
  const scores = computeScores(props);
  const liv = scores.livability;
  const act = scores.activity;

  // Livability grade
  const livCircle = document.getElementById('grade-liv-circle');
  const livLetter = document.getElementById('grade-liv-letter');
  const livSub = document.getElementById('grade-liv-sub');
  livCircle.style.background = gradeColor(liv.grade);
  livLetter.textContent = liv.grade;
  livSub.textContent = `Score: ${liv.score}/100 -- ${liv.present.length} of ${ESSENTIAL_CATEGORIES.length} essentials`;

  // Activity grade
  const actCircle = document.getElementById('grade-act-circle');
  const actLetter = document.getElementById('grade-act-letter');
  const actSub = document.getElementById('grade-act-sub');
  actCircle.style.background = gradeColor(act.grade);
  actLetter.textContent = act.grade;
  actSub.textContent = `Score: ${act.score}/100`;

  // Essentials checklist
  const checklistEl = document.getElementById('checklist-essentials');
  checklistEl.innerHTML = '';
  for (const cat of ESSENTIAL_CATEGORIES) {
    const count = props[cat] || 0;
    const present = count >= 1;
    const item = document.createElement('div');
    item.className = 'checklist-item';
    item.innerHTML = `
      <span class="checklist-icon ${present ? 'present' : 'missing'}">${present ? '&#10003;' : '&#10005;'}</span>
      <span>${CATEGORIES[cat]}</span>
      <span class="checklist-count">${count}</span>
    `;
    checklistEl.appendChild(item);
  }

  // Lifestyle breakdown
  const lifestyleEl = document.getElementById('checklist-lifestyle');
  lifestyleEl.innerHTML = '';
  const maxLifestyle = Math.max(1, ...LIFESTYLE_CATEGORIES.map(c => props[c] || 0));
  for (const cat of LIFESTYLE_CATEGORIES) {
    const count = props[cat] || 0;
    const pct = Math.round((count / maxLifestyle) * 100);
    const item = document.createElement('div');
    item.className = 'lifestyle-item';
    item.innerHTML = `
      <span style="min-width:80px;font-size:12px;color:var(--text-dim)">${CATEGORIES[cat]}</span>
      <div class="lifestyle-bar-wrap">
        <div class="lifestyle-bar" style="width:${pct}%;background:${CATEGORY_COLORS[cat]}"></div>
      </div>
      <span class="lifestyle-count">${count}</span>
    `;
    lifestyleEl.appendChild(item);
  }

  document.getElementById('score-card').classList.add('visible');
}

// ============================================================
// LOADING / UI HELPERS
// ============================================================
function setLoading(visible, text) {
  const el = document.getElementById('loading');
  if (text) document.getElementById('loading-text').textContent = text;
  el.classList.toggle('visible', visible);
}

function showError(msg) {
  const t = document.getElementById('error-toast');
  if (!t) return;
  t.textContent = msg;
  t.style.display = 'block';
  setTimeout(() => { t.style.display = 'none'; }, 4000);
}

// ============================================================
// BUILD CATEGORY CHIPS
// ============================================================
function buildChips() {
  const essGrid = document.getElementById('chips-essentials');
  const lifeGrid = document.getElementById('chips-lifestyle');
  const otherGrid = document.getElementById('chips-other');

  function makeChip(cat, container) {
    const chip = document.createElement('div');
    chip.className = 'cat-chip';
    chip.dataset.cat = cat;
    chip.style.setProperty('--chip-color', CATEGORY_COLORS[cat]);
    chip.innerHTML = `<span class="chip-dot"></span>${CATEGORIES[cat]}`;
    chip.setAttribute('role', 'button');
    chip.setAttribute('tabindex', '0');
    chip.addEventListener('click', () => toggleCategory(cat, chip));
    chip.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        chip.click();
      }
    });
    container.appendChild(chip);
  }

  ESSENTIAL_CATEGORIES.forEach(c => makeChip(c, essGrid));
  LIFESTYLE_CATEGORIES.forEach(c => makeChip(c, lifeGrid));
  OTHER_CATEGORIES.forEach(c => makeChip(c, otherGrid));
}

function buildLayersSheet() {
  const grids = {
    essentials: document.getElementById('layers-essentials'),
    lifestyle: document.getElementById('layers-lifestyle'),
    other: document.getElementById('layers-other'),
  };
  if (!grids.essentials) return;

  function addLayerBtn(cat, container) {
    const btn = document.createElement('button');
    btn.className = 'layer-btn';
    btn.dataset.cat = cat;
    const color = CATEGORY_COLORS[cat] || 'var(--accent)';
    btn.innerHTML = `<span class="layer-dot" style="background:${color}"></span>${CATEGORIES[cat]}`;
    btn.addEventListener('click', () => {
      const sidebarChip = document.querySelector(`.cat-chip[data-cat="${cat}"]`);
      if (sidebarChip) toggleCategory(cat, sidebarChip);
      // Update active states in sheet
      document.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('active'));
      if (activeCategory === cat) btn.classList.add('active');
      // Auto-close sheet after selection
      closeLayersSheet();
    });
    container.appendChild(btn);
  }

  ESSENTIAL_CATEGORIES.forEach(c => addLayerBtn(c, grids.essentials));
  LIFESTYLE_CATEGORIES.forEach(c => addLayerBtn(c, grids.lifestyle));
  OTHER_CATEGORIES.forEach(c => addLayerBtn(c, grids.other));

  // Mode toggle in sheet
  document.querySelectorAll('.layers-mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.dataset.mode;
      const desktopPill = document.querySelector(`.mode-pill[data-mode="${mode}"]`);
      if (desktopPill) desktopPill.click();
      document.querySelectorAll('.layers-mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });

  // FAB opens sheet
  document.getElementById('layers-fab').addEventListener('click', openLayersSheet);
  document.getElementById('layers-sheet-close').addEventListener('click', closeLayersSheet);
  document.getElementById('layers-overlay').addEventListener('click', closeLayersSheet);
}

function openLayersSheet() {
  document.getElementById('layers-sheet').style.display = 'block';
  document.getElementById('layers-overlay').style.display = 'block';
  // Sync active state
  document.querySelectorAll('.layer-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.cat === activeCategory);
  });
  requestAnimationFrame(() => {
    document.getElementById('layers-sheet').classList.add('visible');
  });
}

function closeLayersSheet() {
  const sheet = document.getElementById('layers-sheet');
  sheet.classList.remove('visible');
  setTimeout(() => {
    sheet.style.display = 'none';
    document.getElementById('layers-overlay').style.display = 'none';
  }, 300);
}

function initCityPills() {
  document.querySelectorAll('.city-pill, .mobile-city-bar .mcb-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      const lat = parseFloat(pill.dataset.lat);
      const lon = parseFloat(pill.dataset.lon);
      const name = pill.dataset.name || '';
      map.flyTo({ center: [lon, lat], zoom: 12, duration: 1500 });
      if (typeof umami !== 'undefined') umami.track('city-jump', { city: name });
    });
  });
}

function toggleCategory(cat, chipEl) {
  // If already active, deselect -> return to composite
  if (activeCategory === cat) {
    activeCategory = null;
    document.querySelectorAll('.cat-chip').forEach(c => c.classList.remove('active'));
    removePoisLayer();
    const prop = currentScoreMode === 'activity' ? '_activity' : '_livability';
    updateHexLayerColor(compositeColorExpr(prop));
    updateLegend(currentScoreMode);
    updateUrlState();
    return;
  }

  // Select this category
  activeCategory = cat;
  document.querySelectorAll('.cat-chip').forEach(c => c.classList.remove('active'));
  chipEl.classList.add('active');

  // Update hex coloring based on category type
  if (isEssentialCategory(cat)) {
    updateHexLayerColor(completenessColorExpr(cat));
    updateLegend('essential', cat);
  } else if (isLifestyleCategory(cat)) {
    updateHexLayerColor(densityColorExpr(cat));
    updateLegend('density', cat);
  } else {
    updateHexLayerColor(densityColorExpr(cat));
    updateLegend('density', cat);
  }

  // Load POI dots
  loadPois(cat);
  updateUrlState();
  if (typeof umami !== 'undefined') umami.track('category-select', {category: cat});
}

// ============================================================
// LEGEND
// ============================================================
function updateLegend(mode, cat) {
  const titleEl = document.getElementById('legend-title');
  const barEl = document.getElementById('legend-bar');
  const labelsEl = document.querySelector('.legend-labels');

  if (mode === 'essential') {
    titleEl.textContent = CATEGORIES[cat] || 'Essential';
    barEl.innerHTML = '<span style="background:#374151"></span><span style="background:#1a9850"></span>';
    labelsEl.innerHTML = '<span>Missing</span><span>Present</span>';
  } else if (mode === 'density') {
    titleEl.textContent = CATEGORIES[cat] || 'Density';
    barEl.innerHTML = '<span style="background:#fee5d9"></span><span style="background:#fcae91"></span><span style="background:#fb6a4a"></span><span style="background:#de2d26"></span><span style="background:#a50f15"></span>';
    labelsEl.innerHTML = '<span>Low</span><span>High</span>';
  } else if (mode === 'activity') {
    titleEl.textContent = 'Activity';
    barEl.innerHTML = '<span style="background:#a50026"></span><span style="background:#d73027"></span><span style="background:#f46d43"></span><span style="background:#fdae61"></span><span style="background:#fee08b"></span><span style="background:#a6d96a"></span><span style="background:#1a9850"></span><span style="background:#006837"></span>';
    labelsEl.innerHTML = '<span>F</span><span>A+</span>';
  } else {
    titleEl.textContent = 'Livability';
    barEl.innerHTML = '<span style="background:#a50026"></span><span style="background:#d73027"></span><span style="background:#f46d43"></span><span style="background:#fdae61"></span><span style="background:#fee08b"></span><span style="background:#a6d96a"></span><span style="background:#1a9850"></span><span style="background:#006837"></span>';
    labelsEl.innerHTML = '<span>F</span><span>A+</span>';
  }
}

// ============================================================
// MODE PILLS (Livability / Activity)
// ============================================================
function initModePills() {
  document.querySelectorAll('.mode-pill').forEach(pill => {
    pill.setAttribute('role', 'radio');
    const isActive = pill.dataset.mode === currentScoreMode;
    pill.setAttribute('aria-checked', isActive ? 'true' : 'false');
    if (isActive) pill.classList.add('active');
    else pill.classList.remove('active');

    pill.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        pill.click();
      }
    });

    pill.addEventListener('click', () => {
      currentScoreMode = pill.dataset.mode;
      document.querySelectorAll('.mode-pill').forEach(p => {
        p.classList.remove('active');
        p.setAttribute('aria-checked', 'false');
      });
      pill.classList.add('active');
      pill.setAttribute('aria-checked', 'true');

      // Deselect any active category chip
      if (activeCategory) {
        activeCategory = null;
        document.querySelectorAll('.cat-chip').forEach(c => c.classList.remove('active'));
        removePoisLayer();
      }

      const prop = currentScoreMode === 'activity' ? '_activity' : '_livability';
      updateHexLayerColor(compositeColorExpr(prop));
      updateLegend(currentScoreMode);

      const descEl = document.getElementById('mode-desc');
      if (descEl) {
        descEl.textContent = currentScoreMode === 'livability'
          ? 'Coverage of daily essentials: grocery, healthcare, transit, parks, schools'
          : 'Concentration of dining, nightlife, culture, shopping, sports';
      }

      saveSettings();
    });
  });
}

// ============================================================
// BASEMAP SWITCHER
// ============================================================
function initBasemaps() {
  document.querySelectorAll('.bm-btn').forEach(btn => {
    if (btn.dataset.bm === currentBasemap) btn.classList.add('active');

    btn.addEventListener('click', function() {
      const bm = this.dataset.bm;
      if (!BASEMAPS[bm]) return;

      document.querySelectorAll('.bm-btn').forEach(b => b.classList.remove('active'));
      this.classList.add('active');
      currentBasemap = bm;
      saveSettings();
      updateUrlState();
      if (typeof umami !== 'undefined') umami.track('basemap-switch', {basemap: bm});

      map.setStyle(BASEMAPS[bm].style);
      // Wait for new style to fully load, then restore hex layers
      let _restored = false;
      const restoreHex = () => {
        if (_restored) return;
        if (!map.isStyleLoaded()) {
          setTimeout(restoreHex, 100);
          return;
        }
        _restored = true;
        hexTileCache.clear();
        addHexSource();
        loadHexTiles().then(() => renderHexFeatures());
        if (activeCategory) loadPois(activeCategory);
      };
      setTimeout(restoreHex, 500);
    });
  });
}

// ============================================================
// SIDEBAR TOGGLE
// ============================================================
function initSidebar() {
  const sidebar = document.getElementById('sidebar');
  const toggle = document.getElementById('sidebar-toggle');

  toggle.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
    toggle.innerHTML = sidebar.classList.contains('collapsed') ? '&#9654;' : '&#9664;';
  });

  // Mobile drag handle
  const drag = document.getElementById('sidebar-drag');
  if (drag) {
    drag.addEventListener('click', () => {
      sidebar.classList.toggle('expanded');
    });
  }
}

// ============================================================
// SCORE CARD CLOSE
// ============================================================
document.getElementById('score-card-close').addEventListener('click', () => {
  const card = document.getElementById('score-card');
  card.classList.remove('visible', 'expanded');
  if (_selectedMarker) { _selectedMarker.remove(); _selectedMarker = null; }
});

// Score card swipe to expand/collapse on mobile
(function() {
  const card = document.getElementById('score-card');
  const drag = document.getElementById('score-card-drag');
  let startY = 0;

  drag.addEventListener('touchstart', (e) => {
    startY = e.touches[0].clientY;
  }, { passive: true });

  drag.addEventListener('touchend', (e) => {
    const endY = e.changedTouches[0].clientY;
    const diff = startY - endY;
    if (diff > 30) {
      // Swipe up -> expand
      card.classList.add('expanded');
    } else if (diff < -30) {
      // Swipe down -> collapse or close
      if (card.classList.contains('expanded')) {
        card.classList.remove('expanded');
        card.scrollTop = 0;
      } else {
        card.classList.remove('visible', 'expanded');
        if (_selectedMarker) { _selectedMarker.remove(); _selectedMarker = null; }
      }
    }
  }, { passive: true });

  // Also tap to toggle
  drag.addEventListener('click', () => {
    card.classList.toggle('expanded');
    if (!card.classList.contains('expanded')) card.scrollTop = 0;
  });
})();

document.getElementById('share-btn').addEventListener('click', () => {
  const lat = window._selectedLat || '';
  const lon = window._selectedLon || '';
  const zoom = window._selectedZoom || 14;
  if (!lat || !lon) return;
  let url = `${location.origin}/map?lat=${lat}&lon=${lon}&zoom=${zoom}&ref=share`;
  // Include active category and mode in share link
  if (typeof activeCategory !== 'undefined' && activeCategory) url += `&category=${activeCategory}`;
  if (typeof currentScoreMode !== 'undefined' && currentScoreMode !== 'livability') url += `&mode=${currentScoreMode}`;
  if (typeof umami !== 'undefined') umami.track('share-link', {lat, lon});
  navigator.clipboard.writeText(url).then(() => {
    const btn = document.getElementById('share-btn');
    const txt = document.getElementById('share-btn-text');
    btn.classList.add('copied');
    txt.textContent = 'Copied!';
    setTimeout(() => { btn.classList.remove('copied'); txt.textContent = 'Share'; }, 2000);
  }).catch(() => {
    prompt('Copy this link:', url);
  });
});

// ============================================================
// AUTO-SELECT HEX at coordinates (reused by search + deep links)
// ============================================================
function autoSelectHex(lat, lon, maxRetries) {
  maxRetries = maxRetries || 10;
  let attempts = 0;
  const trySelect = () => {
    attempts++;
    const features = map.queryRenderedFeatures(
      map.project([lon, lat]),
      { layers: ['hex-fill'] }
    );
    if (features && features.length > 0) {
      if (_selectedMarker) _selectedMarker.remove();
      const markerEl = document.createElement('div');
      markerEl.style.cssText = 'width:32px;height:32px;cursor:pointer;filter:drop-shadow(0 2px 6px rgba(0,0,0,0.4));';
      markerEl.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32"><path d="M16 2C10.5 2 6 6.5 6 12c0 3.5 2.2 7.5 5.3 11.5 1.4 1.8 2.8 3.3 3.8 4.3.3.3.6.5.9.7.3-.2.6-.4.9-.7 1-1 2.4-2.5 3.8-4.3C23.8 19.5 26 15.5 26 12 26 6.5 21.5 2 16 2z" fill="#0F6E56"/><circle cx="16" cy="11.5" r="5.5" fill="#1D9E75" opacity="0.35"/><circle cx="16" cy="11.5" r="2.8" fill="#E1F5EE"/></svg>';
      _selectedMarker = new maplibregl.Marker({ element: markerEl, anchor: 'bottom' })
        .setLngLat([lon, lat]).addTo(map);
      window._selectedLat = lat.toFixed(5);
      window._selectedLon = lon.toFixed(5);
      window._selectedZoom = Math.round(map.getZoom());
      showScoreCard(features[0].properties);
      reverseGeocodeAddress(lat, lon);
    } else if (attempts < maxRetries) {
      setTimeout(trySelect, 500);
    }
  };
  setTimeout(trySelect, 500);
}

// ============================================================
// SEARCH (Nominatim)
// ============================================================
let searchTimer = null;

function initSearch() {
  const input = document.getElementById('search-input');
  const results = document.getElementById('search-results');

  input.addEventListener('input', () => {
    clearTimeout(searchTimer);
    const q = input.value.trim();
    if (q.length < 3) {
      results.classList.remove('visible');
      return;
    }
    searchTimer = setTimeout(() => searchNominatim(q), 1000);
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      results.classList.remove('visible');
      input.blur();
    }
  });

  document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-wrap')) {
      results.classList.remove('visible');
    }
  });
}

async function searchNominatim(query) {
  const results = document.getElementById('search-results');
  try {
    const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=10&viewbox=-25,72,45,34&bounded=0&accept-language=en,*`;
    const resp = await fetch(url, {
      headers: { 'Accept-Language': 'en' },
    });
    const rawData = await resp.json();
    // Filter out countries/continents, keep cities and admin areas
    const excludeTypes = new Set(['country', 'continent', 'ocean', 'sea']);
    const data = rawData
      .filter(r => !excludeTypes.has(r.type) && (r.place_rank || 0) >= 6)
      .slice(0, 5);

    results.innerHTML = '';
    if (data.length === 0) {
      results.innerHTML = '<div class="search-result-item" style="color:var(--text-dim)">No results found</div>';
      results.classList.add('visible');
      return;
    }

    for (const item of data) {
      const el = document.createElement('div');
      el.className = 'search-result-item';
      // Show "City, Country" -- add "(province)" for non-city admin areas
      const parts = item.display_name.split(',').map(s => s.trim());
      let label = parts.length > 1 ? `${parts[0]}, ${parts[parts.length - 1]}` : parts[0];
      if ((item.place_rank || 0) <= 12) label += ' (region)';
      el.textContent = label;
      el.addEventListener('click', () => {
        const sLat = parseFloat(item.lat);
        const sLon = parseFloat(item.lon);
        const rank = item.place_rank || 0;
        const isCity = rank <= 15;
        const sZoom = isCity ? 12 : 15;
        // Close any open score card and remove marker before navigating
        document.getElementById('score-card').classList.remove('visible', 'expanded');
        if (_selectedMarker) { _selectedMarker.remove(); _selectedMarker = null; }
        map.flyTo({ center: [sLon, sLat], zoom: sZoom, duration: 1500 });
        results.classList.remove('visible');
        document.getElementById('search-input').value = item.display_name.split(',')[0];
        if (typeof umami !== 'undefined') umami.track('search', {query: item.display_name.split(',')[0]});
        // Auto-select hex at the searched location
        map.once('moveend', () => { autoSelectHex(sLat, sLon, 15); });
      });
      results.appendChild(el);
    }
    results.classList.add('visible');
  } catch (err) {
    console.warn('Search error:', err);
  }
}

// ============================================================
// MAP EVENTS
// ============================================================
map.on('load', () => {
  addHexSource();

  // Deep linking: read lat/lon/zoom/category from URL params (from SEO pages)
  const _params = new URLSearchParams(location.search);
  if (_params.get('lat') && _params.get('lon')) {
    const _lat = parseFloat(_params.get('lat'));
    const _lon = parseFloat(_params.get('lon'));
    const _zoom = parseInt(_params.get('zoom') || '12');
    if (!isNaN(_lat) && !isNaN(_lon)) {
      map.jumpTo({ center: [_lon, _lat], zoom: _zoom });
    }
  }

  // Switch mode from URL param (livability/activity)
  if (_params.get('mode') && _params.get('mode') !== currentScoreMode) {
    const modeBtn = document.querySelector(`.mode-pill[data-mode="${_params.get('mode')}"]`);
    if (modeBtn) modeBtn.click();
  }

  loadHexTiles();

  // Activate category from URL param after tiles load
  if (_params.get('category')) {
    const _cat = _params.get('category');
    setTimeout(() => {
      const chip = document.querySelector(`[data-cat="${_cat}"]`);
      if (chip) chip.click();
    }, 2000);
  }

  // Deep link: place marker and open score card at shared location
  if (_params.get('lat') && _params.get('lon')) {
    const _dlLat = parseFloat(_params.get('lat'));
    const _dlLon = parseFloat(_params.get('lon'));
    // Wait for hex tiles to render, then auto-select
    setTimeout(() => { autoSelectHex(_dlLat, _dlLon); }, 2000);
  }
});

let _poiReloadTimer = null;
let _moveRenderTimer = null;
map.on('move', () => {
  // Throttle: re-render cached hexagons at most every 100ms during panning.
  // Shows cached tiles entering the viewport without waiting for moveend.
  if (!_moveRenderTimer) {
    _moveRenderTimer = setTimeout(() => {
      _moveRenderTimer = null;
      renderHexFeatures();
    }, 100);
  }
});

map.on('moveend', () => {
  scheduleHexLoad();
  updateUrlState();
  if (activeCategory) {
    clearTimeout(_poiReloadTimer);
    _poiReloadTimer = setTimeout(() => loadPois(activeCategory), 700);
  }
});

map.on('zoomend', () => {
  const tileZ = hexTileZoom();
  if (tileZ === null) {
    document.getElementById('zoom-hint').classList.add('visible');
  } else {
    document.getElementById('zoom-hint').classList.remove('visible');
  }
});

// ============================================================
// INIT
// ============================================================
buildChips();
buildLayersSheet();
initCityPills();
initModePills();
initBasemaps();
initSidebar();
initSearch();

// Set initial legend
updateLegend(currentScoreMode);
}); // end DOMContentLoaded
