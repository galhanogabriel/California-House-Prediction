import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import shapely
import streamlit as st

from joblib import load

from notebooks.src.config import CLEANED_DATA, FINAL_MODEL, GEO_MEDIAN_DATA

@st.cache_data
def load_clean_data():
    return pd.read_parquet(CLEANED_DATA)

@st.cache_data
def load_geo_data():
    gdf_geo = gpd.read_parquet(GEO_MEDIAN_DATA)

    # Explode MultiPolygons into individual polygons
    gdf_geo = gdf_geo.explode(ignore_index=True)

    # Function to check and fix invalid geometries
    def fix_and_orient_geometry(geometry):
        if not geometry.is_valid:
            geometry = geometry.buffer(0)  # Fix invalid geometry
        # Orient the polygon to be counter-clockwise if it's a Polygon or MultiPolygon
        if isinstance(
            geometry, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)
        ):
            geometry = shapely.geometry.polygon.orient(geometry, sign=1.0)
        return geometry

    # Apply the fix and orientation function to geometries
    gdf_geo["geometry"] = gdf_geo["geometry"].apply(fix_and_orient_geometry)

    # Extract polygon coordinates
    def get_polygon_coordinates(geometry):
        return (
            [[[x, y] for x, y in geometry.exterior.coords]]
            if isinstance(geometry, shapely.geometry.Polygon)
            else [
                [[x, y] for x, y in polygon.exterior.coords]
                for polygon in geometry.geoms
            ]
        )

    # Apply the coordinate conversion and store in a new column
    gdf_geo["geometry"] = gdf_geo["geometry"].apply(get_polygon_coordinates)

    return gdf_geo

@st.cache_resource
def load_model():
    return load(FINAL_MODEL)

df = load_clean_data()
gdf_geo = load_geo_data()
model = load_model()

st.title("Housing Price Prediction")

counties = sorted(gdf_geo["name"].unique())

column1, column2 = st.columns(2)

with column1:

    with st.form(key="form"):

        select_county = st.selectbox("Counties", counties)

        longitude = gdf_geo.query("name == @select_county")["longitude"].values
        latitude = gdf_geo.query("name == @select_county")["latitude"].values

        housing_median_age = st.number_input("House Age", value=10, min_value=1, max_value=50)

        total_rooms = gdf_geo.query("name == @select_county")["total_rooms"].values
        total_bedrooms = gdf_geo.query("name == @select_county")["total_bedrooms"].values
        population = gdf_geo.query("name == @select_county")["population"].values
        households = gdf_geo.query("name == @select_county")["households"].values

        median_income = st.slider("Median Income (in US$ 1000s)", 5.0, 100.0, 45.0, 5.0)

        ocean_proximity = gdf_geo.query("name == @select_county")["ocean_proximity"].values

        bins_income = [0, 1.5, 3, 4.5, 6, np.inf]
        median_income_cat = np.digitize(median_income / 10, bins=bins_income)

        rooms_per_household = gdf_geo.query("name == @select_county")["rooms_per_household"].values
        bedrooms_per_room = gdf_geo.query("name == @select_county")["bedrooms_per_room"].values
        population_per_household = gdf_geo.query("name == @select_county")["population_per_household"].values

        model_entry = {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income / 10,
            "ocean_proximity": ocean_proximity,
            "median_income_cat": median_income_cat,
            "rooms_per_household": rooms_per_household,
            "bedrooms_per_room": bedrooms_per_room,
            "population_per_household": population_per_household,
        }

        df_model_entry = pd.DataFrame(model_entry)

        predict_button = st.form_submit_button("Predict Price")

    if predict_button:
        price = model.predict(df_model_entry)
        st.metric(label="Predicted Price: (US$)", value=f"{price[0][0]:.2f}")

with column2:

    view_state = pdk.ViewState(
        latitude=float(latitude[0]),
        longitude=float(longitude[0]),
        zoom=5,
        min_zoom=5,
        max_zoom=15
    )

    polygon_layer = pdk.Layer(
        "PolygonLayer",
        data=gdf_geo[["name", "geometry"]],
        get_polygon="geometry",
        get_fill_color=[0, 0, 255, 100],
        get_line_color=[255, 255, 255],
        get_line_width=50,
        pickable=True,
        auto_highlight=True,
    )
    selected_county = gdf_geo.query("name == @select_county")

    highlight_layer = pdk.Layer(
        "PolygonLayer",
        data=selected_county[["name", "geometry"]],
        get_polygon="geometry",
        get_fill_color=[255, 0, 0, 100],
        get_line_color=[0, 0, 0],
        get_line_width=500,
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {
        "html": "<b>County:</b> {name}",
        "style": {"backgroundColor": "steelblue", "color": "white", "fontsize": "10px"},
    }

    map = pdk.Deck(
        initial_view_state=view_state,
        map_style="light",
        layers=[polygon_layer, highlight_layer],
        tooltip=tooltip
    )

    st.pydeck_chart(map)