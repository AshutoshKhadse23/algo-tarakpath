from gridGeneration.gridGen import *
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
import os


def plot_path_on_map(ax, path_lat_lon, color):
    if not path_lat_lon:
        raise ValueError("Path coordinates are empty")
    
    # Handle tuples with more than two values
    try:
        lats_path = [lat for lat, lon, *_ in path_lat_lon]
        lons_path = [lon for lat, lon, *_ in path_lat_lon]
    except ValueError:
        raise ValueError("Invalid path_lat_lon structure. Ensure it contains (lat, lon) tuples.")
    
    ax.scatter(lons_path, lats_path, c=color, s=5, label='Path Points', edgecolors='none')



# Ensure the shapefile path is correct
shapefile_path = '/Users/ashutoshkhadse/Documents/Flutter routes /nautilus/web_server/ne_10m_land.shp'

if not os.path.exists(shapefile_path):
    raise FileNotFoundError(f"Shapefile not found at: {shapefile_path}")

# Initialize the grid and compute paths
grid = Grid()
start_lat, start_lon = 18.902242497778347, 72.79556612242386
end_lat, end_lon = 21.494038748955347, 114.12468335055716

t = time.time()
path_lat_lon1 = grid.a_star(start_lat, start_lon, end_lat, end_lon, 0, 10, True)
path_lat_lon2 = grid.a_star(start_lat, start_lon, end_lat, end_lon, 0, 10, False)
print('Time taken:', time.time() - t)

# Load the map
india_map = gpd.read_file(shapefile_path)

# Plot the map and paths
fig, ax = plt.subplots(figsize=(12, 8))
india_map.plot(ax=ax, color='lightgrey')
ax.set_xlim([30, 130])  # Longitude range for the Indian subcontinent
ax.set_ylim([-35, 35])  # Latitude range for the Indian subcontinent

plot_path_on_map(ax, path_lat_lon1, 'red')
plot_path_on_map(ax, path_lat_lon2, 'orange')

plt.title('Path Traversal on the Indian Subcontinent')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()

# Optionally, save the paths to a CSV
output_csv = 'path.csv'
with open(output_csv, 'w') as f:
    for path in path_lat_lon1 + path_lat_lon2:
        f.write(f"{path[0]},{path[1]}\n")
print(f"Paths saved to {output_csv}")
