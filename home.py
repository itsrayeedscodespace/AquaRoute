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
from streamlit.components.v1 import iframe

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

def find_optimal_route(start_coords, end_coords, target_distance, coordinates, 
                        variation_radius=1.0, num_variations=20, max_iterations=100):
    """
    Finds an optimized route that maintains connectivity while maximizing plastic collection.
    
    Args:
        start_coords: Tuple of (lon, lat) for starting point
        end_coords: Tuple of (lon, lat) for ending point
        target_distance: Desired route distance in kilometers
        coordinates: List of (lon, lat, concentration) tuples
        variation_radius: Maximum deviation from backbone path (in km)
        num_variations: Number of variation points to try at each step
        max_iterations: Maximum number of optimization iterations
    
    Returns:
        tuple: (route_points, message)
            route_points: List of [lat, lon] points defining the route
            message: String describing the route metrics
    """
    def calculate_direct_path(start, end, num_points=10):
        """Creates a straight line path between start and end points"""
        lats = np.linspace(start[1], end[1], num_points)
        lons = np.linspace(start[0], end[0], num_points)
        return list(zip(lons, lats))
    
    def find_nearby_points(center, radius_km, exclude_points=None):
        """Find all points within radius km of center point"""
        if exclude_points is None:
            exclude_points = set()
            
        nearby = []
        center_lat, center_lon = center[1], center[0]
        
        for lon, lat, conc in coordinates:
            if (lon, lat) in exclude_points:
                continue
                
            dist = geopy_distance.distance(
                (lat, lon), 
                (center_lat, center_lon)
            ).kilometers
            
            if dist <= radius_km:
                nearby.append((lon, lat, conc))
                
        return nearby
    
    def calculate_route_metrics(route):
        """Calculate total distance and plastic concentration for a route"""
        total_distance = sum(
            geopy_distance.distance(
                (route[i][1], route[i][0]),
                (route[i+1][1], route[i+1][0])
            ).kilometers
            for i in range(len(route)-1)
        )
        
        # Calculate total concentration by finding nearest concentration point
        total_concentration = 0
        for point in route:
            nearby = find_nearby_points(point, 0.5)
            if nearby:
                # Take highest concentration if multiple points nearby
                total_concentration += max(p[2] for p in nearby)
                
        return total_distance, total_concentration
    
    # Create initial backbone route
    backbone_points = calculate_direct_path(start_coords, end_coords)
    best_route = [(lon, lat, 0) for lon, lat in backbone_points]
    best_distance, best_concentration = calculate_route_metrics(best_route)
    
    # Optimize route through iterations
    for _ in range(max_iterations):
        improved = False
        
        # Try varying each point except start and end
        for i in range(1, len(best_route) - 1):
            current_point = best_route[i]
            prev_point = best_route[i-1]
            next_point = best_route[i+1]
            
            # Generate variation points
            nearby_points = find_nearby_points(
                current_point,
                variation_radius
            )
            
            if not nearby_points:
                continue
                
            # Score variation points
            variations = []
            for variation in nearby_points:
                # Create temporary route with variation
                temp_route = best_route.copy()
                temp_route[i] = variation
                
                # Calculate metrics
                temp_distance, temp_concentration = calculate_route_metrics(temp_route)
                
                # Check if variation maintains target distance constraint
                if abs(temp_distance - target_distance) <= target_distance * 0.2:
                    # Score based on concentration improvement vs distance increase
                    concentration_improvement = temp_concentration - best_concentration
                    distance_penalty = max(0, temp_distance - best_distance)
                    
                    score = concentration_improvement - (distance_penalty * 0.5)
                    variations.append((score, temp_route, temp_distance, temp_concentration))
            
            # Apply best variation if it improves the route
            if variations:
                variations.sort(reverse=True)
                best_variation = variations[0]
                
                if best_variation[0] > 0:  # If score is positive
                    best_route = best_variation[1]
                    best_distance = best_variation[2]
                    best_concentration = best_variation[3]
                    improved = True
        
        # If no improvements found, stop iterations
        if not improved:
            break
    
    # Convert final route to format needed for visualization
    route_points = [[point[1], point[0]] for point in best_route]
    
    message = (
        f"Route optimized! Total distance: {best_distance:.2f} km\n"
        f"Target distance: {target_distance:.2f} km\n"
        f"Total plastic concentration: {best_concentration:.2f} pieces/m³"
    )
    
    return route_points, message
@st.cache_resource
def load_image():
    image = Image.open("static/img/bgimageocean.jpg")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

@st.cache_resource
def load_video():
    with open("vid/oceanvid.mp4", "rb") as video_file:
        video_bytes = video_file.read()
    return base64.b64encode(video_bytes).decode()

# Load resources
img_str = load_image()
video_str = load_video()

# Custom CSS for the video container and overlay text
st.markdown("""
    <style>
    .video-container {
        position: relative;
        width: 100%;
        height: 70vh;
        overflow: hidden;
    }
    .video-wrapper {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        opacity: 0.3;  /* Adjust this value to change video opacity */
    }
    .video-wrapper video {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .overlay-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: white;
        font-size: 3vw;
        font-weight: bold;
        text-align: center;
        width: 100%;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        z-index: 1;
    }
    </style>
    """, unsafe_allow_html=True)

# Display video with overlay text
st.markdown(f"""
    <div class="video-container">
        <div class="video-wrapper">
            <video autoplay loop muted playsinline>
                <source src="data:video/mp4;base64,{video_str}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        <div class="overlay-text">
            Welcome to AquaRoute<br>
            <span style="font-size: 35px;">Plastic Cleanup Route Optimizer</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Add some spacing after the video
st.markdown("<br>", unsafe_allow_html=True)

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

st.title("AquaRoute - Plastic Cleanup Route Optimizer")
st.subheader("Plan Your Route")

col1, col2 = st.columns([2, 1])

with col1:
    # Determine the map center based on selected positions
    if st.session_state['stop_position']:
        map_center = list(map(float, st.session_state['stop_position'].split(',')))[::-1]
    elif st.session_state['start_position']:
        map_center = list(map(float, st.session_state['start_position'].split(',')))[::-1]
    else:
        map_center = st.session_state['map_center']

    m = folium.Map(location=map_center, zoom_start=5)
    
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
                st.session_state['map_center'] = [clicked_lat, clicked_lon]
                st.rerun()
            else:
                st.error("Selected point is on land. Please select an ocean coordinate.")
        elif not st.session_state['stop_position']:
            if is_valid_ocean_point(clicked_lat, clicked_lon):
                st.session_state['stop_position'] = f"{clicked_lon:.6f}, {clicked_lat:.6f}"
                st.session_state['map_center'] = [clicked_lat, clicked_lon]
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
            st.write(f"Minimum distance (direct route): {10*min_distance:.2f} km")
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
        st.session_state['distance'] = ''
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

                with st.spinner('Optimizing route...'):
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

        # Calculate impact metrics
        total_concentration = sum(st.session_state['route_points'])
        plastic_bottles = int(total_concentration)
        plastic_waste_tons = round((plastic_bottles/0.035)/1000, 2)
        shipping_containers = max(1, int(plastic_waste_tons / 30))
        years_to_decompose = 400

        col_text, col_graph = st.columns([1, 1])

        with col_text:
            # Create impact statement
            impact_html = f"""
            <div style="background-color: black; padding: 20px; border-radius: 10px; margin-top: 20px;">
                <h3 style="color: #1c5d99; margin-bottom: 15px;">Route Impact Analysis</h3>
                <p style="font-size: 18px; margin-bottom: 30px;">
                    This route passes through waters containing the equivalent of:
                </p>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li style="font-size: 20px; margin-bottom: 8px;">
                        <span style="color: #1c5d99; font-weight: bold; font-size: 40px;">🥤 {plastic_bottles:,}</span>&nbsp;&nbsp;&nbsp;plastic bottles
                    </li>
                    <li style="font-size: 20px; margin-bottom: 8px;">
                        <span style="color: #1c5d99; font-weight: bold; font-size: 40px;">🗑️ {plastic_waste_tons:,}</span>&nbsp;&nbsp;&nbsp;tons of plastic waste
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

        with col_graph:
            # Create bar graph
            fig = go.Figure()
            
            colors = ['#00b4d8', '#0077be', '#023e8a']
            
            fig.add_trace(go.Bar(
                x=['Plastic Bottles', 'Plastic Waste (tons)', 'Shipping Containers'],
                y=[plastic_bottles, plastic_waste_tons, shipping_containers],
                marker_color=colors,
                text=[f'{plastic_bottles:,}', f'{plastic_waste_tons:,}', f'{shipping_containers}'],
                textposition='outside',
                textfont=dict(size=12, color='white'),
                hoverinfo='y+name',
                hovertemplate='%{y:,.0f}<extra></extra>',
            ))

            fig.update_layout(
                title={
                    'text': 'Environmental Impact Metrics',
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=20, color='white')
                },
                xaxis_title=None,
                yaxis_title=None,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis_type='log',
                height=500,
                margin=dict(l=50, r=20, t=80, b=20),
                showlegend=False,
            )

            fig.update_xaxes(
                showgrid=False,
                showline=True,
                linecolor='rgba(255,255,255,0.3)',
                tickfont=dict(size=12, color='white'),
                tickangle=0
            )

            fig.update_yaxes(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                showline=True,
                linecolor='rgba(255,255,255,0.3)',
                zeroline=False,
                tickfont=dict(size=12, color='white'),
                tickformat='.2s'
            )

            st.plotly_chart(fig, use_container_width=True)


st.markdown("---")  # This adds a horizontal line for separation

st.subheader("Global Microplastic Distribution")

# Display the heatmap image
try:
    image = Image.open("img/heatmap.png")
    st.image(image, use_column_width=True, caption="Global Microplastic Distribution Heatmap")
except FileNotFoundError:
    st.error("Error: The heatmap image was not found. Please check if it exists in the img folder.")

st.markdown("<br>", unsafe_allow_html=True)  # Add some space

st.subheader("Further Reading")

st.write("""
This research article provides comprehensive data on microplastic abundance in the world's upper oceans and Great Lakes. 
It offers valuable insights into the current state of microplastic pollution and can help inform future research and cleanup efforts.
""")

# Embed the PDF
def show_pdf(file_path):
    with open(file_path,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

try:
    show_pdf("study.pdf")
except FileNotFoundError:
    st.error("Error: The PDF file was not found. Please check if it exists in the correct location.")

# Provide a download link for the PDF
try:
    with open("study.pdf", "rb") as file:
        btn = st.download_button(
            label="Download PDF",
            data=file,
            file_name="microplastic_study.pdf",
            mime="application/pdf"
        )
except FileNotFoundError:
    st.error("Error: The PDF file was not found. Please check if it exists in the correct location.")
