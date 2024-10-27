import streamlit as st
import folium
from streamlit_folium import st_folium
from PIL import Image
import base64
from io import BytesIO
import numpy as np
from geopy import distance as geopy_distance
import pickle
from global_land_mask import globe
from scipy.spatial import KDTree
import heapq
from folium import plugins
import csv
import plotly.graph_objs as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="AquaRoute", layout="wide")

@st.cache_resource
def load_data():
    # with open("zdata.pickle", "rb") as file:
    #     x, y, concentration = pickle.loads(file.read())
    with open("griddy.csv", "r") as f:
        reader = csv.DictReader(f)
        data = list(reader)
    x = []
    y = []
    concentration = []
    for point in data:   
        x.append(float(point["longitude"]))
        y.append(float(point["latitude"]))
        concentration.append(float(point["Level_0_(pieces/m3)"]))
    for i in range(len(concentration)):
        concentration[i] = abs(concentration[i])
    return x, y, concentration

def is_valid_ocean_point(lat, lon):
    return not globe.is_land(lat, lon)

def is_ocean_path(point1, point2, num_checks=10):
    """Check if the path between two points crosses land"""
    lats = np.linspace(point1[0], point2[0], num_checks)
    lons = np.linspace(point1[1], point2[1], num_checks)
    
    for lat, lon in zip(lats, lons):
        if not is_valid_ocean_point(lat, lon):
            return False
    return True

def heuristic(point1, point2):
    # Ensure we're only using lat and lon for distance calculation
    return geopy_distance.distance(point1[:2][::-1], point2[:2][::-1]).kilometers

def find_optimal_route(start_coords, end_coords, target_distance, coordinates, tree, plastic_weight=0.8):
    # Add input validation at the beginning of the function
    if not (-180 <= start_coords[0] <= 180) or not (-90 <= start_coords[1] <= 90):
        return None, "Invalid start coordinates. Longitude must be between -180 and 180, latitude between -90 and 90."
    if not (-180 <= end_coords[0] <= 180) or not (-90 <= end_coords[1] <= 90):
        return None, "Invalid end coordinates. Longitude must be between -180 and 180, latitude between -90 and 90."

    if not (is_valid_ocean_point(start_coords[1], start_coords[0]) and is_valid_ocean_point(end_coords[1], end_coords[0])):
        return None, "Start or end point is on land. Please select ocean coordinates."

    # start and end are already in (lon, lat) format, no need to convert
    start = start_coords
    end = end_coords

    # Find closest points in our dataset
    _, start_idx = tree.query([start[0], start[1]])
    _, end_idx = tree.query([end[0], end[1]])
    
    start_point = coordinates[start_idx]
    end_point = coordinates[end_idx]
    
    direct_distance = heuristic(start_point, end_point)
    if target_distance < direct_distance:
        return None, f"Target distance ({target_distance:.2f} km) is less than minimum possible distance ({direct_distance:.2f} km)."

    open_set = []
    heapq.heappush(open_set, (0, 0, start_point))
    
    came_from = {}
    g_score = {start_point: 0}
    plastic_score = {start_point: start_point[2]}
    
    visited = set()
    
    while open_set:
        current_priority, current_g, current = heapq.heappop(open_set)
        
        if current in visited:
            continue
            
        visited.add(current)
        
        if current == end_point:
            path = []
            total_plastic = 0
            route_points = []
            curr = current
            while curr != start_point:
                path.append(curr)
                total_plastic += curr[2]
                route_points.append(curr[2])
                curr = came_from[curr]
            path.append(start_point)
            route_points.append(start_point[2])
            path.reverse()
            route_points.reverse()
            
            # Convert path coordinates to (lat, lon) format for Folium
            folium_path = [(point[1], point[0]) for point in path]
            
            # Verify entire path is through ocean
            for i in range(len(folium_path) - 1):
                if not is_ocean_path(folium_path[i], folium_path[i + 1]):
                    continue  # Skip this path and keep searching
            
            st.session_state['route_points'] = route_points
            return folium_path, f"Found route with total waste concentration score: {total_plastic:.2f}"
        
        k = min(30, len(coordinates))
        distances, indices = tree.query([current[0], current[1]], k=k)
        
        if not isinstance(distances, (list, tuple, np.ndarray)):
            distances = [distances]
            indices = [indices]
            
        for dist, index in zip(distances, indices):
            neighbor = coordinates[index]
            
            if neighbor in visited:
                continue
            
            # Check if the path to neighbor crosses land
            if not is_ocean_path((current[1], current[0]), (neighbor[1], neighbor[0])):
                continue
                
            tentative_g_score = g_score[current] + heuristic(current, neighbor)
            
            if tentative_g_score > target_distance:
                continue
            
            tentative_plastic_score = plastic_score[current] + neighbor[2]
            
            distance_to_end = heuristic(neighbor, end_point)
            path_deviation = abs(tentative_g_score - direct_distance) / direct_distance
            
            plastic_density = tentative_plastic_score / tentative_g_score
            normalized_distance = distance_to_end / direct_distance
            
            priority = (
                (1 - plastic_weight) * normalized_distance +
                (0.1 * path_deviation) -
                (plastic_weight * 2 * plastic_density)
            )
            
            if neighbor not in g_score or tentative_plastic_score > plastic_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                plastic_score[neighbor] = tentative_plastic_score
                heapq.heappush(open_set, (priority, tentative_g_score, neighbor))
    
    return None, "No valid path found meeting the distance constraints. Try adjusting the target distance."

# Initialize session state
if 'start_position' not in st.session_state:
    st.session_state['start_position'] = ''
if 'stop_position' not in st.session_state:
    st.session_state['stop_position'] = ''
if 'distance' not in st.session_state:
    st.session_state['distance'] = ''
if 'map_center' not in st.session_state:
    st.session_state['map_center'] = [27.5, -140.0]
if 'optimized_route' not in st.session_state:
    st.session_state['optimized_route'] = None
if 'route_message' not in st.session_state:
    st.session_state['route_message'] = ''
if 'show_optimized_map' not in st.session_state:
    st.session_state['show_optimized_map'] = False
if 'route_points' not in st.session_state:
    st.session_state['route_points'] = []

# Load data and create KDTree
try:
    lon, lat, concentration = load_data()
    # Create coordinates list for KDTree
    coordinates = list(zip(lon, lat, concentration))
    points = [(x, y) for x, y, _ in coordinates]
    tree = KDTree(points)
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

st.title("AquaRoute - Ocean Route Optimizer")
st.subheader("Plan Your Route")

col1, col2 = st.columns([2, 1])

with col1:
    m = folium.Map(location=st.session_state['map_center'], zoom_start=3)
    
    # Add markers for start and stop positions if they exist
    if st.session_state['start_position']:
        start_lon, start_lat = map(float, st.session_state['start_position'].split(','))
        folium.Marker(
            [start_lat, start_lon],
            popup="Start",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(m)
    
    if st.session_state['stop_position']:
        stop_lon, stop_lat = map(float, st.session_state['stop_position'].split(','))
        folium.Marker(
            [stop_lat, stop_lon],
            popup="Stop",
            icon=folium.Icon(color="red", icon="stop")
        ).add_to(m)
    
    map_data = st_folium(m, width=700, height=500, key="input_map")

    if map_data['last_clicked'] is not None:
        clicked_lat = map_data['last_clicked']['lat']
        clicked_lon = map_data['last_clicked']['lng']
        
        if not st.session_state['start_position']:
            if is_valid_ocean_point(clicked_lat, clicked_lon):
                st.session_state['start_position'] = f"{clicked_lon:.6f}, {clicked_lat:.6f}"
                st.rerun()
            else:
                st.error("Selected point is on land. Please select an ocean coordinate.")
        elif not st.session_state['stop_position']:
            if is_valid_ocean_point(clicked_lat, clicked_lon):
                st.session_state['stop_position'] = f"{clicked_lon:.6f}, {clicked_lat:.6f}"
                st.rerun()
            else:
                st.error("Selected point is on land. Please select an ocean coordinate.")

with col2:
    st.write("### Route Parameters")
    st.write("Click on the map to select points or enter coordinates manually.")
    
    start_position = st.text_input("Start Position (lon, lat)", 
                                   value=st.session_state['start_position'], 
                                   key='start_position_input',
                                   help="Format: longitude, latitude")
    
    stop_position = st.text_input("Stop Position (lon, lat)", 
                                  value=st.session_state['stop_position'], 
                                  key='stop_position_input',
                                  help="Format: longitude, latitude")
    
    if start_position and stop_position:
        try:
            start_coords = tuple(map(float, start_position.split(',')))
            stop_coords = tuple(map(float, stop_position.split(',')))
            min_distance = geopy_distance.distance(start_coords[::-1], stop_coords[::-1]).kilometers
            st.write(f"Minimum distance (direct route): {min_distance:.2f} km")
            st.write("Enter a target distance greater than the minimum.")
        except ValueError:
            st.error("Invalid coordinate format. Please use 'longitude, latitude'.")
    
    distance = st.text_input("Target Distance (km)", 
                             value=st.session_state['distance'], 
                             key='distance_input',
                             help="Desired route distance in kilometers")

    st.session_state['start_position'] = start_position
    st.session_state['stop_position'] = stop_position
    st.session_state['distance'] = distance

    if st.button("Clear Points"):
        st.session_state['start_position'] = ''
        st.session_state['stop_position'] = ''
        st.session_state['optimized_route'] = None
        st.session_state['route_message'] = ''
        st.session_state['show_optimized_map'] = False
        st.rerun()
    if st.button("Optimize Route"):
        st.session_state['show_optimized_map'] = True
        
        if not (start_position and stop_position and distance):
            st.error("Please provide start position, stop position, and target distance.")
        else:
            try:
                # Parse start coordinates
                start_coords = tuple(map(float, start_position.strip().split(',')))
                if len(start_coords) != 2:
                    raise ValueError("Start position must have exactly 2 values (longitude, latitude)")

                # Parse end coordinates
                stop_coords = tuple(map(float, stop_position.strip().split(',')))
                if len(stop_coords) != 2:
                    raise ValueError("End position must have exactly 2 values (longitude, latitude)")

                # Parse distance
                target_distance = float(distance)
                if target_distance <= 0:
                    raise ValueError("Distance must be greater than 0")

                # Calculate minimum distance
                min_distance = geopy_distance.distance(start_coords[::-1], stop_coords[::-1]).kilometers
                if target_distance < min_distance:
                    raise ValueError(f"Target distance ({target_distance:.2f} km) is less than minimum possible distance ({min_distance:.2f} km).")

                optimized_route, message = find_optimal_route(
                    start_coords, 
                    stop_coords, 
                    target_distance,
                    coordinates,
                    tree
                )
                
                st.session_state['optimized_route'] = optimized_route
                st.session_state['route_message'] = message
                
            except ValueError as e:
                st.error(f"Input error: {str(e)}")
            except Exception as e:
                st.error(f"Error during route optimization: {str(e)}")

if st.session_state['show_optimized_map']:
    st.subheader("Optimized Route")

    col3, col4 = st.columns([2, 1])

    with col3:
        # Create a new map without heatmap for the optimized route
        m = folium.Map(location=st.session_state['map_center'], zoom_start=3)
        
        if st.session_state['optimized_route']:
            route = st.session_state['optimized_route']
            
            # Add start marker
            folium.Marker(
                route[0],
                popup="Start",
                icon=folium.Icon(color="green", icon="play")
            ).add_to(m)
            
            # Add end marker
            folium.Marker(
                route[-1],
                popup="Stop",
                icon=folium.Icon(color="red", icon="stop")
            ).add_to(m)
            
            # Add route line
            folium.PolyLine(
                route,
                color='blue',
                weight=3,
                opacity=0.8,
                popup="Optimized Route"
            ).add_to(m)
            
            # Add intermediate waypoints
            for i, point in enumerate(route[1:-1], 1):
                folium.CircleMarker(
                    point,
                    radius=3,
                    color='blue',
                    fill=True,
                    popup=f'Waypoint {i}'
                ).add_to(m)
            
            m.fit_bounds([route[0], route[-1]])
        
        st_folium(m, width=700, height=500, key="optimized_map")

    with col4:
        if st.session_state['optimized_route']:
            st.success(st.session_state['route_message'])
            st.write("### Route Details")
            st.write(f"Number of waypoints: {len(st.session_state['optimized_route'])}")
            st.write(f"Start position: {st.session_state['start_position']}")
            st.write(f"Stop position: {st.session_state['stop_position']}")
            
            # Calculate actual distance
            total_distance = sum(
                geopy_distance.distance(st.session_state['optimized_route'][i], 
                                      st.session_state['optimized_route'][i+1]).kilometers
                for i in range(len(st.session_state['optimized_route'])-1)
            )
            st.write(f"Actual route distance: {total_distance:.2f} km")
            
        elif st.session_state['route_message']:
            st.error(st.session_state['route_message'])


    # After the optimized map section, add this new section for the concentration profile and accumulated concentration graphs
    if st.session_state['optimized_route'] and st.session_state['route_points']:
        st.subheader("Plastic Concentration Analysis")

        # Create subplots
        fig = make_subplots(rows=2, cols=1, 
                            subplot_titles=("Plastic Concentration Profile", "Accumulated Plastic Concentration"),
                            vertical_spacing=0.2)

        # Concentration profile
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(st.session_state['route_points']) + 1)),
                y=st.session_state['route_points'],
                mode='lines+markers',
                name='Concentration',
                line=dict(color='#00b4d8', width=3),
                marker=dict(size=8, color='#0077be', symbol='circle')
            ),
            row=1, col=1
        )

        # Accumulated concentration
        accumulated = np.cumsum(st.session_state['route_points'])
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(accumulated) + 1)),
                y=accumulated,
                mode='lines+markers',
                name='Accumulated',
                line=dict(color='#ff6b6b', width=3),
                marker=dict(size=8, color='#ee5253', symbol='circle')
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            height=900,
            title_text="Plastic Concentration Analysis",
            title_font=dict(size=24, color='#333333'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )

        # Update x-axes
        fig.update_xaxes(
            title_text="Waypoint Number",
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
            row=1, col=1
        )
        fig.update_xaxes(
            title_text="Waypoint Number",
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
            row=2, col=1
        )

        # Update y-axes
        fig.update_yaxes(
            title_text="Plastic Concentration (pieces/m³)",
            showgrid=True,
            gridcolor='rgba(204, 204, 204, 0.5)',
            zeroline=False,
            showline=False,
            showticklabels=True,
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Accumulated Plastic (pieces/m³)",
            showgrid=True,
            gridcolor='rgba(204, 204, 204, 0.5)',
            zeroline=False,
            showline=False,
            showticklabels=True,
            row=2, col=1
        )

        st.plotly_chart(fig, use_container_width=True)

    # Add this new section for the impact statement
    if st.session_state['optimized_route'] and st.session_state['route_points']:
        st.subheader("Environmental Impact")

        # Calculate total concentration
        total_concentration = sum(st.session_state['route_points'])

        # Calculate impact metrics
        plastic_bottles = int(total_concentration)  # Assuming 1 piece ≈ 1 bottle for simplicity
        plastic_waste_tons = round(total_concentration * 0.001, 2)  # Assuming average weight of 1g per piece
        shipping_containers = max(1, int(plastic_waste_tons / 30))  # Assuming 30 tons per container
        years_to_decompose = 400  # Fixed value as per example

        # Create impact statement
        impact_html = f"""
        <div style="background-color: black; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h3 style="color: #1c5d99; margin-bottom: 15px;">Route Impact Analysis</h3>
            <p style="font-size: 18px; margin-bottom: 30px;">
                This route passes through waters containing the equivalent of:
            </p>
            <ul style="list-style-type: none; padding-left: 0;">
                <li style="font-size: 20px; margin-bottom: 8px;">
                    <span style="color: #1c5d99; font-weight: bold; font-size: 40px;">🍶 {plastic_bottles:,}</span>&nbsp;&nbsp;&nbsp;plastic bottles
                </li>
                <li style="font-size: 20px; margin-bottom: 8px;">
                    <span style="color: #1c5d99; font-weight: bold; font-size: 40px;">⚖️ {plastic_waste_tons:,}</span>&nbsp;&nbsp;&nbsp;tons of plastic waste
                </li>
                <li style="font-size: 20px; margin-bottom: 8px;">
                    <span style="color: #1c5d99; font-weight: bold; font-size: 40px;">🚢 {shipping_containers}</span>&nbsp;&nbsp;&nbsp;standard shipping container{'s' if shipping_containers > 1 else ''}
                </li>
                <li style="font-size: 20px; margin-bottom: 8px;">
                    <span style="color: #1c5d99; font-weight: bold; font-size: 40px;">⏳ {years_to_decompose}</span>&nbsp;&nbsp;&nbsp;years to decompose
                </li>
            </ul>
            <p style="font-style: italic; margin-top: 15px;">
                These estimates highlight the importance of ocean cleanup efforts and reducing plastic pollution.
            </p>
        </div>
        """

        st.markdown(impact_html, unsafe_allow_html=True)
