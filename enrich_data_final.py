import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import osmium
import time
import requests
from tqdm import tqdm
import warnings

# Suppress minor pandas fragmentation warnings for clean output
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# =========================================================
# THE OFFLINE MAP PARSER
# =========================================================
class SwissPOIHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.poi_data = {
            "transit": [], "shops": [], "parks": [], 
            "schools": [], "hospitals": [], 
            "lakes": [], "rivers": [],
            "nightlife": [], "noisy_roads": [], 
            "noisy_trains": [], "pedestrian_zones": []
        }
        self.transit_names = []

    def process_tags(self, tags, lat, lon):
        name = tags.get('name', 'Unnamed Stop')
        
        # 1. Transit
        if tags.get('railway') in ['station', 'tram_stop'] or tags.get('highway') == 'bus_stop':
            self.poi_data["transit"].append([lat, lon])
            self.transit_names.append(name)
            
        # 2. Standard POIs
        elif tags.get('shop') == 'supermarket':
            self.poi_data["shops"].append([lat, lon])
        elif tags.get('leisure') == 'park':
            self.poi_data["parks"].append([lat, lon])
        elif tags.get('amenity') == 'school':
            self.poi_data["schools"].append([lat, lon])
        elif tags.get('amenity') in ['hospital', 'clinic']:
            self.poi_data["hospitals"].append([lat, lon])
            
        # 3. Water Features
        elif tags.get('natural') == 'water' or tags.get('water') == 'lake':
            self.poi_data["lakes"].append([lat, lon])
        elif tags.get('waterway') in ['river', 'stream', 'canal']:
            self.poi_data["rivers"].append([lat, lon])
            
        # 4. Nightlife (Vibe Proxy)
        elif tags.get('amenity') in ['restaurant', 'cafe', 'bar', 'pub', 'nightclub']:
            self.poi_data["nightlife"].append([lat, lon])
            
        # 5. Noise Proxies
        elif tags.get('highway') in ['motorway', 'trunk', 'primary']:
            self.poi_data["noisy_roads"].append([lat, lon])
        elif tags.get('railway') == 'rail':
            self.poi_data["noisy_trains"].append([lat, lon])
            
        # 6. Pedestrian Zones (Walkability Proxy)
        elif tags.get('highway') in ['pedestrian', 'living_street']:
            self.poi_data["pedestrian_zones"].append([lat, lon])

    def node(self, n):
        try: self.process_tags(n.tags, n.location.lat, n.location.lon)
        except osmium.InvalidLocationError: pass

    def way(self, w):
        try:
            mid_idx = len(w.nodes) // 2
            loc = w.nodes[mid_idx].location
            self.process_tags(w.tags, loc.lat, loc.lon)
        except (osmium.InvalidLocationError, IndexError): pass


def build_ultimate_dataset():
    # --- FILE PATHS ---
    csv_path = "raw_data/sred_data_withmontageimages_latlong.csv"
    pbf_path = "raw_data/switzerland-260417.osm.pbf"
    crime_csv = "raw_data/ktzh_00001202_00003600.csv"
    
    # --- COLUMN NAMES (Update if your CSV uses different names) ---
    lat_col, lon_col = 'geo_lat', 'geo_lng'
    price_col = 'price' 
    rooms_col = 'number_of_rooms' 
    city_col = 'object_city' 

    print("\n[START] Loading Main Dataset...")
    df = pd.read_csv(csv_path)
    valid_mask = df[lat_col].notna() & df[lon_col].notna()
    
    # =========================================================
    # PART 1: OFFLINE OPENSTREETMAP PARSING
    # =========================================================
    print(f"\n--- PART 1: OFFLINE SPATIAL ANALYSIS ---")
    print(f"Parsing map file: {pbf_path}")
    start_time = time.time()
    
    handler = SwissPOIHandler()
    handler.apply_file(pbf_path, locations=True) 
    
    print(f"✅ Map parsed in {round(time.time() - start_time, 2)} seconds!")

    apt_coords = np.radians(df.loc[valid_mask, [lat_col, lon_col]].values)
    EARTH_RADIUS = 6371000  

    # -- 500m Counts --
    radius_500m = 500 / EARTH_RADIUS 
    count_cats = ['transit', 'shops', 'parks', 'schools', 'hospitals', 'nightlife', 'pedestrian_zones']

    for cat in tqdm(count_cats, desc="Calculating 500m Densities"):
        coords = handler.poi_data[cat]
        col_name = f'{cat}_count_500m'
        df[col_name] = 0
        if coords:
            tree = BallTree(np.radians(coords), metric='haversine')
            df.loc[valid_mask, col_name] = tree.query_radius(apt_coords, r=radius_500m, count_only=True)

    # -- Exact Distances --
    all_cats = count_cats + ['lakes', 'rivers', 'noisy_roads', 'noisy_trains']

    for cat in tqdm(all_cats, desc="Calculating Exact Distances"):
        coords = handler.poi_data[cat]
        dist_col = f'dist_to_{cat}_m'
        df[dist_col] = 999999 
        
        if coords:
            tree = BallTree(np.radians(coords), metric='haversine')
            distances, indices = tree.query(apt_coords, k=1)
            df.loc[valid_mask, dist_col] = np.round(distances.flatten() * EARTH_RADIUS, 0)
            
            if cat == 'transit':
                names_array = np.array(handler.transit_names)
                df.loc[valid_mask, 'nearest_transit_name'] = names_array[indices.flatten()]

    # =========================================================
    # PART 2: GRANULAR AREA RENT CALCULATOR (By City & Rooms)
    # =========================================================
    print(f"\n--- PART 2: REAL ESTATE ECONOMICS ---")
    if all(col in df.columns for col in [price_col, city_col, rooms_col]):
        df['avg_rent_for_room_type'] = df.groupby([city_col, rooms_col])[price_col].transform('mean').round(0)
        df['rent_vs_average'] = df[price_col] - df['avg_rent_for_room_type']
        print(f"✅ Calculated 'Deal Score' (Rent vs Average) by City and Room Count.")
    else:
        print(f"⚠️ Missing columns for rent calculation. Skipping.")

    # =========================================================
    # PART 3: TOPOGRAPHIC ELEVATION VIA API
    # =========================================================
    print(f"\n--- PART 3: TOPOGRAPHY (ELEVATION) ---")
    df['elevation_m'] = np.nan
    valid_lats = df.loc[valid_mask, lat_col].tolist()
    valid_lons = df.loc[valid_mask, lon_col].tolist()
    elevations = []

    chunk_size = 100
    for i in tqdm(range(0, len(valid_lats), chunk_size), desc="Fetching Bulk Elevation"):
        chunk_lats = valid_lats[i:i+chunk_size]
        chunk_lons = valid_lons[i:i+chunk_size]
        
        url = "https://api.open-meteo.com/v1/elevation"
        params = {"latitude": ",".join(map(str, chunk_lats)), "longitude": ",".join(map(str, chunk_lons))}
        try:
            resp = requests.get(url, params=params).json()
            elevations.extend(resp.get("elevation", [np.nan]*len(chunk_lats)))
        except:
            elevations.extend([np.nan]*len(chunk_lats))
            time.sleep(1)

    df.loc[valid_mask, 'elevation_m'] = elevations
    print(f"✅ Added precise elevation data.")

    # =========================================================
    # PART 4: WEIGHTED CRIME SAFETY INDEX
    # =========================================================
    print(f"\n--- PART 4: SAFETY ANALYTICS ---")
    try:
        crime_df = pd.read_csv(crime_csv)
        latest_year = crime_df['Ausgangsjahr'].max()
        crime_latest = crime_df[crime_df['Ausgangsjahr'] == latest_year].copy()
        
        # Severity Weights
        weights = {
            'Total gegen Leib und Leben': 10.0, 'Total gegen die Freiheit': 10.0,
            'Total gegen sexuelle Integrität': 10.0, 'Total schwere Fälle': 8.0,
            'Total gegen das Vermögen': 5.0, 'Total leichte Fälle': 2.0,
            'Total übrige Titel StGB': 2.0, 'Total Übertretungen': 1.0
        }
        crime_latest['severity_weight'] = crime_latest['Haupttitel'].map(weights).fillna(0.1)
        crime_latest['adjusted_crimes'] = crime_latest['Straftaten_total'] * crime_latest['severity_weight']
        
        # Aggregate by City
        city_crime = crime_latest.groupby('Gemeindename')['adjusted_crimes'].sum().reset_index()
        city_crime.rename(columns={'adjusted_crimes': 'weighted_crime_score'}, inplace=True)
        
        # Normalize per 1000 residents
        pop_latest = crime_latest.groupby('Gemeindename')['Einwohner'].max().reset_index()
        city_crime = city_crime.merge(pop_latest, on='Gemeindename')
        city_crime['weighted_crime_per_1000'] = (city_crime['weighted_crime_score'] / city_crime['Einwohner']) * 1000
        
        # Merge into main Dataset
        df = df.merge(city_crime[['Gemeindename', 'weighted_crime_per_1000']], 
                      left_on=city_col, right_on='Gemeindename', how='left')
        df.drop(columns=['Gemeindename'], inplace=True, errors='ignore')
        
        # Fill NaN with Median
        df['weighted_crime_per_1000'] = df['weighted_crime_per_1000'].fillna(df['weighted_crime_per_1000'].median()).round(1)
        print(f"✅ Built and integrated Weighted Safety Index.")
        
    except FileNotFoundError:
        print(f"⚠️ Could not find {crime_csv}. Skipping crime data.")

    # =========================================================
    # SAVE THE FINAL DATASET
    # =========================================================
    print(f"\n[FINISH] Saving the Ultimate Dataset...")
    df.to_csv(csv_path, index=False)
    print("🚀 SUCCESS! Your Datathon model is fully armed and ready.")

if __name__ == "__main__":
    build_ultimate_dataset()