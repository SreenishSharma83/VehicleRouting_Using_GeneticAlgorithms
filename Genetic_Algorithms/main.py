import streamlit as st
import numpy as np
import pandas as pd
import pulp
import itertools
import gmaps
import googlemaps
import matplotlib.pyplot as plt

# Streamlit configuration
st.set_page_config(page_title="CVRP Solver", page_icon=":truck:")

# Google Maps API configuration
API_KEY = 'YOUR_GOOGLE_MAPS_API_KEY'
gmaps.configure(api_key=API_KEY)
googlemaps_client = googlemaps.Client(key=API_KEY)

# Title
st.title("Capacitated Vehicle Routing Problem Solver")

# Inputs
st.sidebar.header("Inputs")
customer_count = st.sidebar.number_input("Number of Customers", min_value=2, value=10)
vehicle_count = st.sidebar.number_input("Number of Vehicles", min_value=1, value=4)
vehicle_capacity = st.sidebar.number_input("Vehicle Capacity", min_value=1, value=50)

# Fix random seed
np.random.seed(seed=777)

# Set depot latitude and longitude
depot_latitude = 40.748817
depot_longitude = -73.985428

# Generate customer data
df = pd.DataFrame({"latitude": np.random.normal(depot_latitude, 0.007, customer_count),
                   "longitude": np.random.normal(depot_longitude, 0.007, customer_count),
                   "demand": np.random.randint(10, 20, customer_count)})

# Set the depot location and demand to 0
df.iloc[0, 0] = depot_latitude
df.iloc[0, 1] = depot_longitude
df.iloc[0, 2] = 0

# Function to plot locations on Google Maps
def plot_on_gmaps(_df):
    _marker_locations = []
    for i in range(len(_df)):
        _marker_locations.append((_df['latitude'].iloc[i], _df['longitude'].iloc[i]))
    
    _fig = gmaps.figure()
    _markers = gmaps.marker_layer(_marker_locations)
    _fig.add_layer(_markers)
    return _fig

# Function to calculate distances between locations
def distance_calculator(_df):
    _distance_result = np.zeros((len(_df), len(_df)))
    _df['latitude-longitude'] = '0'
    for i in range(len(_df)):
        _df['latitude-longitude'].iloc[i] = str(_df.latitude[i]) + ',' + str(_df.longitude[i])
    
    for i in range(len(_df)):
        for j in range(len(_df)):
            _google_maps_api_result = googlemaps_client.directions(_df['latitude-longitude'].iloc[i],
                                                                   _df['latitude-longitude'].iloc[j],
                                                                   mode='driving')
            _distance_result[i][j] = _google_maps_api_result[0]['legs'][0]['distance']['value']
    return _distance_result

# Calculate distances
distance = distance_calculator(df)

# Solve the CVRP problem
solution_found = False
for v_count in range(1, vehicle_count + 1):
    problem = pulp.LpProblem("CVRP", pulp.LpMinimize)
    x = [[[pulp.LpVariable(f"x{i}_{j},{k}", cat="Binary") if i != j else None for k in range(v_count)] 
          for j in range(customer_count)] for i in range(customer_count)]
    
    # Objective function
    problem += pulp.lpSum(distance[i][j] * x[i][j][k] if i != j else 0
                          for k in range(v_count) 
                          for j in range(customer_count) 
                          for i in range(customer_count))
    
    # Constraints
    for j in range(1, customer_count):
        problem += pulp.lpSum(x[i][j][k] if i != j else 0 
                              for i in range(customer_count) 
                              for k in range(v_count)) == 1 

    for k in range(v_count):
        problem += pulp.lpSum(x[0][j][k] for j in range(1, customer_count)) == 1
        problem += pulp.lpSum(x[i][0][k] for i in range(1, customer_count)) == 1

    for k in range(v_count):
        for j in range(customer_count):
            problem += pulp.lpSum(x[i][j][k] if i != j else 0 
                                  for i in range(customer_count)) -  pulp.lpSum(x[j][i][k] for i in range(customer_count)) == 0

    for k in range(v_count):
        problem += pulp.lpSum(df.demand[j] * x[i][j][k] if i != j else 0 for i in range(customer_count) for j in range(1, customer_count)) <= vehicle_capacity 

    subtours = []
    for i in range(2, customer_count):
        subtours += itertools.combinations(range(1, customer_count), i)

    for s in subtours:
        problem += pulp.lpSum(x[i][j][k] if i != j else 0 for i, j in itertools.permutations(s, 2) for k in range(v_count)) <= len(s) - 1

    if problem.solve() == 1:
        solution_found = True
        vehicle_count_solution = v_count
        total_distance = pulp.value(problem.objective)
        break

if solution_found:
    st.sidebar.write(f"Vehicle Requirements: {vehicle_count_solution}")
    st.sidebar.write(f"Total Distance: {total_distance}")

    # Plotting on Google Maps
    st.header("Map Visualization")
    plot_result = plot_on_gmaps(df)
    st.pyplot(plot_result)

    # Matplotlib visualization
    st.header("Matplotlib Visualization")
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(customer_count):    
        if i == 0:
            ax.scatter(df.latitude[i], df.longitude[i], c='green', s=200)
            ax.text(df.latitude[i], df.longitude[i], "depot", fontsize=12)
        else:
            ax.scatter(df.latitude[i], df.longitude[i], c='orange', s=200)
            ax.text(df.latitude[i], df.longitude[i], str(df.demand[i]), fontsize=12)

    for k in range(vehicle_count_solution):
        for i in range(customer_count):
            for j in range(customer_count):
                if i != j and pulp.value(x[i][j][k]) == 1:
                    ax.plot([df.latitude[i], df.latitude[j]], [df.longitude[i], df.longitude[j]], c="black")

    st.pyplot(fig)
else:
    st.write("No feasible solution found with the given parameters.")
