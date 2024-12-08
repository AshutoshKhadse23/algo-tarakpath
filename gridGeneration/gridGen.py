import csv
import math
import numpy as np
from gridGeneration.gridGenUtils import *
from sklearn.neighbors import BallTree
from models.cell import GridCell
import heapq
import time
import aiohttp
import logging

GRID_CELL_SIZE_KM = 10
EARTH_RADIUS_KM = 6371.0

# Cache for weather data to avoid redundant API calls
weather_data_cache = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Grid:
    use_a_star = False
    grid = None
    all_points = []
    grid_cells = []
    point_to_ind = {}
    kdtree = None
    num_cols_per_row = []
    coord_to_bathy = {}
    wgrid = None
    def __init__(self) -> None:
        # setting up the grid 
        #doing all the reqired things like setting up the grid and creating neighbour trees and search indexes
        t = time.time()
        self.grid,self.all_points, self.grid_cells, self.point_to_ind, self.num_cols_per_row, self.coord_to_bathy,self.wgrid  = generate_grid()

        #setting up the nearest neighbour tree
        # self.ball_tree = set_up_nearest_neighbour_tree(self.all_points)
        self.kdtree = build_KDTree(self.all_points)
        p = time.time()

        print("Time took to load :", p - t)

    # Function to fetch weather data with caching
    async def get_weather_data(self, lat, lon):
        async with aiohttp.ClientSession() as session:
            key = (round(lat, 4), round(lon, 4))
            if key in weather_data_cache:
                logger.info(f"Using cached weather data for ({lat}, {lon})")
                return weather_data_cache[key]

            logger.info(f"Fetching weather data for coordinates ({lat}, {lon})...")
            url = 'https://api.open-meteo.com/v1/forecast'
            params = {'latitude': lat, 'longitude': lon}
            try:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    weather_data_cache[key] = data
                    logger.info(f"Weather data received for ({lat}, {lon}).")
                    return data
            except Exception as e:
                logger.info(f"Error fetching weather data: {e}")
                return None
        
    # Function to simulate ocean currents
    def get_current_data(self, lat, lon):
        logger.info(f"Simulating ocean current data for ({lat}, {lon})...")
        current_speed = abs(math.sin(math.radians(lat))) * 2  # Simulated speed
        current_direction = lon % 360
        return {
            'current_speed': current_speed,
            'current_direction': current_direction
        }
    
    # Function to calculate fuel efficiency adjustment based on weather and currents
    def fuel_efficiency_adjustment(self, weather_data, current_data):
        adjustment = 1.0
        logger.info('Calculating fuel efficiency adjustment')

        # Apply weather thresholds to adjust cost
        if weather_data:
            if 'temperature_2m' in weather_data:
                temp = weather_data['temperature_2m']
                if temp < 0 or temp > 35:
                    logger.info(f"Temperature out of optimal range: {temp}°C")
                    adjustment += 0.1  # Penalty for extreme temperatures

            if 'precipitation_probability' in weather_data:
                precipitation_prob = weather_data['precipitation_probability']
                if precipitation_prob > 30:
                    logger.info(f"High precipitation probability: {precipitation_prob}%")
                    adjustment += 0.2  # Penalty for high chance of rain

            if 'windspeed_10m' in weather_data:
                wind_speed = weather_data['windspeed_10m']
                if wind_speed > 15:
                    logger.info(f"High wind speed: {wind_speed} m/s")
                    adjustment += 0.15  # Penalty for strong winds

            if 'visibility' in weather_data:
                visibility = weather_data['visibility']
                if visibility < 1000:
                    logger.info(f"Low visibility: {visibility} m")
                    adjustment += 0.2  # Penalty for low visibility

            if 'cloudcover_total' in weather_data:
                cloud_cover = weather_data['cloudcover_total']
                if cloud_cover > 70:
                    logger.info(f"High cloud cover: {cloud_cover}%")
                    adjustment += 0.1  # Penalty for heavy cloud cover

        # Adjust for ocean current speed
        if current_data:
            current_speed = current_data['current_speed']
            adjustment -= current_speed * 0.01  # Favorable currents reduce cost
            logger.info(f"Adjusted for current speed: {current_speed} m/s, new adjustment factor: {adjustment}")

        logger.info(f"Final fuel efficiency adjustment factor: {adjustment}")
        return adjustment

    def get_nearest_cell(self, lat, lon):

        distance, index = get_nearest_kdtree_node(self.kdtree, lat, lon)
        return self.grid_cells[index[0][0]] , distance[0][0] * EARTH_RADIUS_KM
    

    def dijkstra(self, start_lat, start_lon, end_lat, end_lon):
        # Find the nearest start and end cells
        start_cell, _ = self.get_nearest_cell(start_lat, start_lon)
        end_cell, _ = self.get_nearest_cell(end_lat, end_lon)

        start = (start_cell.lat, start_cell.lon)
        end = (end_cell.lat, end_cell.lon)
        logger.info(start)
        logger.info(end)
        # Get indices of start and end cells
        start_idx = self.point_to_ind.get(start, None)
        end_idx = self.point_to_ind.get(end, None)

        if start_idx is None or end_idx is None:
            raise ValueError("Start or end cell not found in grid.")

        start_row, start_col = start_idx
        end_row, end_col = end_idx

        logger.info(self.grid[start_row][start_col].is_land)
        logger.info(self.grid[end_row][end_col].is_land)

        # Initialize distances and previous cell tracking
        num_rows = len(self.grid)
        distances = np.full((num_rows, max(self.num_cols_per_row)), np.inf)
        distances[start_row][start_col] = 0
        prev = np.full((num_rows, max(self.num_cols_per_row)), None)

        priority_queue = []
        heapq.heappush(priority_queue, (0, (start_row, start_col)))
        traversed = []
        while priority_queue:
            current_distance, (row, col) = heapq.heappop(priority_queue)

            traversed.append((self.grid[row][col].lat,self.grid[row][col].lon))
            # Check if we reached the goal
            if(row == end_row and col == end_col):
                break

            # if current_distance > distances[row][col]:
            #     continue

            # Consider all 8 possible directions
            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < num_rows and 0 <= c < self.num_cols_per_row[r]:
                    neighbor = self.grid[r][c]
                    if neighbor is not None and not neighbor.is_land and neighbor.bathymetry_depth <= -20 and not neighbor.near_coastline:
                        distance = haversine(self.grid[row][col].lat, self.grid[row][col].lon,neighbor.lat, neighbor.lon)

                        # distance = abs(self.grid[row][col].lat - self.grid[r][c].lat) * 111.32 + abs(self.grid[row][col].lon - self.grid[r][c].lon) * 111.32

                        new_distance = current_distance + distance
                        if new_distance < distances[r][c]:
                            distances[r][c] = new_distance
                            prev[r][c] = (row, col)
                            heapq.heappush(priority_queue, (new_distance, (r, c)))

        # Reconstruct the shortest path
        path = [] 
        step = (end_row, end_col)
        while step is not None:
            path.append(step)
            step = prev[step[0]][step[1]]
        path.reverse()
        logger.info("size:", path.__len__())
        logger.info("Total Distance Taken:", distances[end_row][end_col])
        # Convert indices to lat/lon for path
        path_lat_lon = [(self.grid[row][col].lat, self.grid[row][col].lon) for row, col in path]

        return (path_lat_lon,traversed)

    async def a_star(self, start_lat, start_lon, end_lat, end_lon, initial_time, initial_speed, flag, session):
        # Initial setup (keeping existing code)
        start_lat = float(start_lat)
        start_lon = float(start_lon)
        end_lat = float(end_lat)
        end_lon = float(end_lon)
        initial_speed = float(initial_speed)
        
        start_cell, dis = self.get_nearest_cell(start_lat, start_lon)
        end_cell, dis = self.get_nearest_cell(end_lat, end_lon)
        
        start = (start_cell.lat, start_cell.lon)
        end = (end_cell.lat, end_cell.lon)
        
        # Define directions (keeping existing directions array)
        directions = [
            (0, 3), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (3, 0), (3, -1),
            (3, -2), (3, -3), (2, -3), (1, -3), (0, -3), (-1, -3), (-2, -3),
            (-3, -3), (-3, -2), (-3, -1), (-3, 0), (-3, 1), (-3, 2), (-3, 3),
            (-2, 3), (-1, 3)
        ]
        
        angles = np.radians(np.arange(0, 360, 15))
        
        start_idx = self.point_to_ind.get(start, None)
        end_idx = self.point_to_ind.get(end, None)
        
        if start_idx is None or end_idx is None:
            raise ValueError("Start or end cell not found in grid.")
        
        start_row, start_col = start_idx
        end_row, end_col = end_idx
        
        # Initialize data structures
        num_rows = len(self.grid)
        distances = np.full((num_rows, max(self.num_cols_per_row)), np.inf)
        distances[start_row][start_col] = 0
        prev = np.full((num_rows, max(self.num_cols_per_row)), None)
        
        priority_queue = []
        heapq.heappush(priority_queue, (0, (start_row, start_col, initial_time)))
        end_time = 0
        
        while priority_queue:
            current_priority, (row, col, curr_time) = heapq.heappop(priority_queue)
            
            if haversine(self.grid[row][col].lat, self.grid[row][col].lon, end_lat, end_lon) < 30:
                end_row = row
                end_col = col
                break
                
            if (row == end_row and col == end_col):
                break
                
            for i, (dr, dc) in enumerate(directions):
                r, c = row + dr, col + dc
                if 0 <= r < num_rows and 0 <= c < self.num_cols_per_row[r]:
                    neighbor = self.grid[r][c]
                    if neighbor is not None and not neighbor.is_land and neighbor.bathymetry_depth <= -20 and not neighbor.near_coastline:
                        # Get weather data for the neighbor cell
                        logger.info(neighbor.lat, neighbor.lon)
                        weather_data = await self.get_weather_data(neighbor.lat, neighbor.lon)
                        
                        # Get ocean current data
                        current_data = self.get_current_data(neighbor.lat, neighbor.lon)
                        
                        # Calculate fuel efficiency adjustment
                        efficiency_factor = self.fuel_efficiency_adjustment(weather_data, current_data)
                        
                        # Calculate base distance
                        distance = haversine(
                            self.grid[row][col].lat, self.grid[row][col].lon,
                            neighbor.lat, neighbor.lon
                        )
                        
                        # Adjust distance based on efficiency factor
                        adjusted_distance = distance * efficiency_factor
                        
                        # Calculate time considering adjusted speed due to conditions
                        effective_speed = initial_speed / efficiency_factor
                        dtime = (distance/effective_speed) * (5/18)
                        
                        new_distance = distances[row][col] + adjusted_distance
                        heuristic = haversine(neighbor.lat, neighbor.lon, 
                                            self.grid[end_row][end_col].lat, 
                                            self.grid[end_row][end_col].lon)
                        
                        end_time = curr_time + dtime
                        
                        # Get wave data cost from existing implementation
                        wave_cost = get_cost(r, c, angles[i], end_time, self.wgrid)
                        
                        # Combine all factors into final priority
                        # Adjusting weights: 60% for distance/heuristic, 20% for weather/current effects, 20% for wave conditions
                        priority = (
                            0.6 * (heuristic + new_distance) + 
                            0.2 * (efficiency_factor * distance) +
                            0.2 * wave_cost
                        )
                        
                        if new_distance < distances[r][c]:
                            distances[r][c] = new_distance
                            prev[r][c] = (row, col, end_time)
                            heapq.heappush(priority_queue, (priority, (r, c, end_time)))
        
        # Reconstruct path (keeping existing code)
        logger.info("entT", end_time)
        end_time = int(end_time)
        wcell = self.wgrid[end_row][end_col]
        path_lat_lon = [[end[0], end[1], wcell.Thgt[(end_time + 1)%24], 
                        wcell.Tper[(end_time + 1)%24], wcell.Tdir[(end_time + 1)%24]]]
        
        s = (end_row, end_col, end_time)
        while s is not None:
            gcell = self.grid[s[0]][s[1]]
            wcell = self.wgrid[s[0]][s[1]]
            time = int(s[2]) % 24
            path_lat_lon.append((gcell.lat, gcell.lon, wcell.Thgt[time], 
                                wcell.Tper[time], wcell.Tdir[time]))
            s = prev[s[0]][s[1]]
        
        path_lat_lon.reverse()
        logger.info("Total Distance Taken:", distances[end_row][end_col])
        return path_lat_lon
