import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import osmium
import time
import requests
from tqdm import tqdm
import warnings
import glob
import os

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
        if tags.get('railway') in ['station', 'tram_stop'] or tags.get('highway') == 'bus_stop':
            self.poi_data["transit"].append([lat, lon])
            self.transit_names.append(name)
        elif tags.get('shop') == 'supermarket':
            self.poi_data["shops"].append([lat, lon])
        elif tags.get('leisure') == 'park':
            self.poi_data["parks"].append([lat, lon])
        elif tags.get('amenity') == 'school':
            self.poi_data["schools"].append([lat, lon])
        elif tags.get('amenity') in ['hospital', 'clinic']:
            self.poi_data["hospitals"].append([lat, lon])
        elif tags.get('natural') == 'water' or tags.get('water') == 'lake':
            self.poi_data["lakes"].append([lat, lon])
        elif tags.get('waterway') in ['river', 'stream', 'canal']:
            self.poi_data["rivers"].append([lat, lon])
        elif tags.get('amenity') in ['restaurant', 'cafe', 'bar', 'pub', 'nightclub']:
            self.poi_data["nightlife"].append([lat, lon])
        elif tags.get('highway') in ['motorway', 'trunk', 'primary']:
            self.poi_data["noisy_roads"].append([lat, lon])
        elif tags.get('railway') == 'rail':
            self.poi_data["noisy_trains"].append([lat, lon])
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


# =========================================================
# GLOBAL PRE-PROCESSING HELPERS
# =========================================================
def prep_crime_data(crime_csv):
    print(f"\n[PREP] Loading Canton Zürich Crime Data...")
    try:
        crime_df = pd.read_csv(crime_csv)
        latest_year = crime_df['Ausgangsjahr'].max()
        crime_latest = crime_df[crime_df['Ausgangsjahr'] == latest_year].copy()
        
        weights = {
            'Total gegen Leib und Leben': 10.0, 'Total gegen die Freiheit': 10.0,
            'Total gegen sexuelle Integrität': 10.0, 'Total schwere Fälle': 8.0,
            'Total gegen das Vermögen': 5.0, 'Total leichte Fälle': 2.0,
            'Total übrige Titel StGB': 2.0, 'Total Übertretungen': 1.0
        }
        crime_latest['severity_weight'] = crime_latest['Haupttitel'].map(weights).fillna(0.1)
        crime_latest['adjusted_crimes'] = crime_latest['Straftaten_total'] * crime_latest['severity_weight']
        
        # Grouping by Gemeinde (Municipality) within Canton Zürich
        city_crime = crime_latest.groupby('Gemeindename')['adjusted_crimes'].sum().reset_index()
        city_crime.rename(columns={'adjusted_crimes': 'weighted_crime_score'}, inplace=True)
        
        pop_latest = crime_latest.groupby('Gemeindename')['Einwohner'].max().reset_index()
        city_crime = city_crime.merge(pop_latest, on='Gemeindename')
        city_crime['weighted_crime_per_1000'] = (city_crime['weighted_crime_score'] / city_crime['Einwohner']) * 1000
        
        # --- NEW: Min-Max Normalization (0 to 100) ---
        c_min = city_crime['weighted_crime_per_1000'].min()
        c_max = city_crime['weighted_crime_per_1000'].max()
        
        # Prevent division by zero if all municipalities had the exact same crime rate
        if c_max > c_min:
            city_crime['crime_index_0_100'] = ((city_crime['weighted_crime_per_1000'] - c_min) / (c_max - c_min)) * 100
        else:
            city_crime['crime_index_0_100'] = 0.0
        # ---------------------------------------------

        return city_crime[['Gemeindename', 'weighted_crime_per_1000', 'crime_index_0_100']]
    except FileNotFoundError:
        print(f"⚠️ Crime data not found at {crime_csv}. Returning None.")
        return None

def build_spatial_trees(handler):
    print("\n[PREP] Building KD-Trees for Spatial Lookups (Done Once)...")
    trees = {}
    for cat, coords in tqdm(handler.poi_data.items(), desc="Compiling Trees"):
        if coords:
            trees[cat] = BallTree(np.radians(coords), metric='haversine')
        else:
            trees[cat] = None
    return trees

# =========================================================
# SINGLE CSV PROCESSING PIPELINE
# =========================================================
def process_single_csv(file_path, handler, trees, crime_data_df):
    print(f"\n--- Processing: {os.path.basename(file_path)} ---")
    
    lat_col, lon_col = 'geo_lat', 'geo_lng'
    price_col, rooms_col, area_col, city_col = 'price', 'number_of_rooms', 'object_zip', 'object_city'
    EARTH_RADIUS = 6371000  

    df = pd.read_csv(file_path)

    # List of all possible columns this script generates
    cols_to_drop = [
        'avg_rent_for_room_type', 'rent_vs_average', 'elevation_m',
        'weighted_crime_per_1000', 'crime_index_0_100', 'nearest_transit_name'
    ]
    # Add all distance and count column names to the drop list
    count_cats = ['transit', 'shops', 'parks', 'schools', 'hospitals', 'nightlife', 'pedestrian_zones']
    all_cats = count_cats + ['lakes', 'rivers', 'noisy_roads', 'noisy_trains']
    
    cols_to_drop.extend([f'{cat}_count_500m' for cat in count_cats])
    cols_to_drop.extend([f'dist_to_{cat}_m' for cat in all_cats])

    # Drop them if they exist in the dataframe before processing
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors='ignore')

    if lat_col not in df.columns or lon_col not in df.columns:
        print(f"⚠️ Skipping {file_path}: Missing lat/lon columns.")
        return

    valid_mask = df[lat_col].notna() & df[lon_col].notna()
    apt_coords = np.radians(df.loc[valid_mask, [lat_col, lon_col]].values)

    # 1. 500m Counts
    radius_500m = 500 / EARTH_RADIUS 
    count_cats = ['transit', 'shops', 'parks', 'schools', 'hospitals', 'nightlife', 'pedestrian_zones']
    for cat in count_cats:
        col_name = f'{cat}_count_500m'
        df[col_name] = 0
        if trees[cat]:
            df.loc[valid_mask, col_name] = trees[cat].query_radius(apt_coords, r=radius_500m, count_only=True)

    # 2. Exact Distances
    all_cats = count_cats + ['lakes', 'rivers', 'noisy_roads', 'noisy_trains']
    for cat in all_cats:
        dist_col = f'dist_to_{cat}_m'
        df[dist_col] = 999999 
        if trees[cat]:
            distances, indices = trees[cat].query(apt_coords, k=1)
            df.loc[valid_mask, dist_col] = np.round(distances.flatten() * EARTH_RADIUS, 0)
            if cat == 'transit':
                names_array = np.array(handler.transit_names)
                df.loc[valid_mask, 'nearest_transit_name'] = names_array[indices.flatten()]

    # 3. Real Estate Deal Score
    if all(col in df.columns for col in [price_col, area_col, rooms_col]):
        df['avg_rent_for_room_type'] = df.groupby([area_col, rooms_col])[price_col].transform('mean').round(0)
        df['rent_vs_average'] = df[price_col] - df['avg_rent_for_room_type']

    # 5. Merge Crime Data
    if crime_data_df is not None and city_col in df.columns:
        df = df.merge(crime_data_df, left_on=city_col, right_on='Gemeindename', how='left')
        df.drop(columns=['Gemeindename'], inplace=True, errors='ignore')
        
        df['weighted_crime_per_1000'] = df['weighted_crime_per_1000'].round(1)
        # --- NEW: Round the new normalized index ---
        df['crime_index_0_100'] = df['crime_index_0_100'].round(1)

    # 6. Save back to the file
    save_path = file_path.replace(".csv", "_enriched.csv")
    df.to_csv(save_path, index=False)
    print(f"✅ Saved enriched data to: {save_path}")


# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    # Define your paths
    pbf_path = "raw_data/additional/switzerland-260417.osm.pbf"
    crime_csv = "raw_data/additional/ktzh_00001202_00003600.csv"
    data_folder = "raw_data" 
    
    # Grab all target CSV files in the folder (excluding the crime csv itself and previously enriched files)
    all_csvs = [f for f in glob.glob(os.path.join(data_folder, "*.csv")) 
                if "ktzh" not in f and "_enriched" not in f]

    if not all_csvs:
        print("No CSV files found to process. Check your folder path.")
        exit()

    print(f"Found {len(all_csvs)} dataset(s) to enrich.")

    # 1. Parse Crime Data Once
    crime_data_df = prep_crime_data(crime_csv)

    # 2. Parse Map Data Once
    print(f"\n[PREP] Parsing offline map file: {pbf_path}")
    start_time = time.time()
    handler = SwissPOIHandler()
    handler.apply_file(pbf_path, locations=True) 
    print(f"✅ Map parsed in {round(time.time() - start_time, 2)} seconds!")

    # 3. Build spatial lookups Once
    trees = build_spatial_trees(handler)

    # 4. Iterate and Process each CSV
    for csv_file in all_csvs:
        process_single_csv(csv_file, handler, trees, crime_data_df)
        
    print("\n🚀 SUCCESS! All datasets are fully enriched and ready.")