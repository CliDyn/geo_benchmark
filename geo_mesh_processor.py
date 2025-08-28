import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import requests
import time
import json
from pathlib import Path

def create_global_mesh(resolution=10):
    """Create a global mesh with specified resolution in degrees"""
    lons = np.arange(-180, 180, resolution)
    lats = np.arange(-60, 85, resolution)
    
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    
    # Flatten to get coordinate pairs
    mesh_points = []
    for i in range(len(lats)):
        for j in range(len(lons)):
            mesh_points.append({
                'lon': float(lon_mesh[i, j]),
                'lat': float(lat_mesh[i, j]),
                'is_land': False,
                'location_info': None,
                'address_full': None,
                'address_formatted': None,
                'country': None,
                'state': None,
                'city': None
            })
    
    return mesh_points, lon_mesh, lat_mesh

def check_land_points(mesh_points, land_shapefile_path):
    """Check which points are on land using geopandas"""
    try:
        land_shp = gpd.read_file(land_shapefile_path)
        
        for point_data in mesh_points:
            lon, lat = point_data['lon'], point_data['lat']
            point = Point(lon, lat)
            is_on_land = land_shp.contains(point).any()
            point_data['is_land'] = bool(is_on_land)
            
        return mesh_points
    except Exception as e:
        print(f"Error loading land shapefile: {e}")
        return mesh_points

def get_location_info(lat, lon):
    """Get location information from Nominatim API"""
    url = "https://nominatim.openstreetmap.org/reverse"
    
    params = {
        "lat": lat,
        "lon": lon,
        "format": "geojson",
        "extratags": 1,
        "namedetails": 1,
        "zoom": 18
    }
    headers = {
        "User-Agent": "climsight",
        "accept-language": "en"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} {response.reason}")
            return None
    except Exception as e:
        print(f"Request error: {e}")
        return None

def parse_address_info(location):
    """Parse address information from Nominatim response"""
    if not location or 'features' not in location or len(location['features']) == 0:
        return None, None, None, None, None
    
    try:
        address = location['features'][0]['properties']['address']
        
        location_str = "Address: "
        for key, value in address.items():
            location_str += f"{key}: {value}, "
        location_str = location_str.rstrip(', ')
        
        location_str_for_print = "**Address:** "
        if "country" in address:
            location_str_for_print += f"{address['country']}, "
        if "state" in address:
            location_str_for_print += f"{address['state']}, "
        if "city" in address:
            location_str_for_print += f"{address['city']}, "
        if "road" in address:
            location_str_for_print += f"{address['road']} "
        if "house_number" in address:
            location_str_for_print += f"{address['house_number']}"
        
        location_str_for_print = location_str_for_print.rstrip(', ')
        country = address.get("country", "")
        state = address.get("state", "")
        city = address.get("city", "")
        
        return location_str, location_str_for_print, country, state, city
    except Exception as e:
        print(f"Error parsing address: {e}")
        return None, None, None, None, None

def process_mesh(resolution=10):
    """Main function to process the global mesh"""
    print(f"Creating global mesh with {resolution}x{resolution} degree resolution...")
    mesh_points, lon_mesh, lat_mesh = create_global_mesh(resolution)
    
    print(f"Created mesh with {len(mesh_points)} points")
    
    # Check for land points
    land_path = './data/land/ne_10m_land.shp'
    print("Checking land points...")
    mesh_points = check_land_points(mesh_points, land_path)
    
    land_points = [p for p in mesh_points if p['is_land']]
    print(f"Found {len(land_points)} land points")
    
    # Get location information for land points
    print("Getting location information for land points...")
    for i, point_data in enumerate(mesh_points):
        if point_data['is_land']:
            print(f"Processing land point {i+1}/{len(land_points)}: ({point_data['lat']}, {point_data['lon']})")
            
            location = get_location_info(point_data['lat'], point_data['lon'])
            if location:
                point_data['location_info'] = location
                
                address_full, address_formatted, country, state, city = parse_address_info(location)
                point_data['address_full'] = address_full
                point_data['address_formatted'] = address_formatted
                point_data['country'] = country
                point_data['state'] = state
                point_data['city'] = city
            
            # Rate limiting - sleep to respect Nominatim terms of use
            time.sleep(1)
    
    return mesh_points, lon_mesh, lat_mesh

def save_mesh_data(mesh_points, lon_mesh, lat_mesh, output_file='mesh_data.json', resolution=10):
    """Save mesh data to file"""
    # Create meshes directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving data to {output_file}...")
    
    # Convert numpy arrays to lists for JSON serialization
    data = {
        'mesh_points': mesh_points,
        'lon_mesh': lon_mesh.tolist(),
        'lat_mesh': lat_mesh.tolist(),
        'mesh_shape': lon_mesh.shape,
        'is_on_regular': True,
        'resolution': resolution,
        'mesh_info': {
            'type': 'regular_grid',
            'resolution_degrees': resolution,
            'lon_range': [-180, 180],
            'lat_range': [-60, 85],
            'total_points': len(mesh_points)
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Also save as CSV for easier reading
    df_data = []
    for point in mesh_points:
        df_data.append({
            'lon': point['lon'],
            'lat': point['lat'],
            'is_land': point['is_land'],
            'country': point['country'] if point['country'] else np.nan,
            'state': point['state'] if point['state'] else np.nan,
            'city': point['city'] if point['city'] else np.nan,
            'address_formatted': point['address_formatted'] if point['address_formatted'] else np.nan,
            'address_full': point['address_full'] if point['address_full'] else np.nan,
            'is_on_regular': True,
            'resolution': resolution
        })
    
    df = pd.DataFrame(df_data)
    csv_filename = output_file.replace('.json', '.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {output_file} and {csv_filename}")

def load_mesh_data(input_file='mesh_data.json'):
    """Load mesh data from file and return all saved parameters"""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    mesh_points = data['mesh_points']
    resolution = data.get('resolution', 10)
    mesh_info = data.get('mesh_info', {})
    
    # Check if this is a chunk file (doesn't have lon_mesh/lat_mesh)
    if 'lon_mesh' in data and 'lat_mesh' in data:
        # Full mesh file
        lon_mesh = np.array(data['lon_mesh'])
        lat_mesh = np.array(data['lat_mesh'])
        mesh_shape = data.get('mesh_shape')
        is_on_regular = data.get('is_on_regular', True)
    else:
        # Chunk file - create minimal arrays for compatibility
        lon_mesh = np.array([])
        lat_mesh = np.array([])
        mesh_shape = None
        is_on_regular = True
    
    return {
        'mesh_points': mesh_points,
        'lon_mesh': lon_mesh,
        'lat_mesh': lat_mesh,
        'mesh_shape': mesh_shape,
        'resolution': resolution,
        'is_on_regular': is_on_regular,
        'mesh_info': mesh_info
    }

if __name__ == "__main__":
    import sys
    
    # Get resolution from command line argument if provided
    resolution = 10  # default
    if len(sys.argv) > 1:
        try:
            resolution = float(sys.argv[1])
        except ValueError:
            print("Invalid resolution value. Using default 10 degrees.")
            resolution = 10
    
    # Process the mesh
    mesh_points, lon_mesh, lat_mesh = process_mesh(resolution)
    
    # Save the data
    save_mesh_data(mesh_points, lon_mesh, lat_mesh, f'meshes/mesh_data_{resolution}deg.json', resolution)
    
    # Print summary
    land_points = [p for p in mesh_points if p['is_land']]
    points_with_info = [p for p in land_points if p['location_info'] is not None]
    
    print(f"\nSummary:")
    print(f"Total points: {len(mesh_points)}")
    print(f"Land points: {len(land_points)}")
    print(f"Land points with location info: {len(points_with_info)}")