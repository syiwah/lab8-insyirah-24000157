
import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Interactive Map Visualization",
    page_icon="🌍",
    layout="wide"
)

st.title("Interactive Map Visualization Examples")
st.write("TFB3133/TEB3133 Data Visualization - Lab 7")

# ==========================================
# 2. LOAD NATURAL EARTH DATASET
# ==========================================

DATA_PATH = Path("ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")


@st.cache_data
def load_data():

    gdf = gpd.read_file(DATA_PATH)

    # Identify country name column
    if "name" in gdf.columns:
        country_col = "name"
    elif "NAME" in gdf.columns:
        country_col = "NAME"
    elif "ADMIN" in gdf.columns:
        country_col = "ADMIN"
    else:
        raise ValueError("Country name column not found.")

    # Get population estimates
    if "POP_EST" in gdf.columns:
        gdf["pop_est"] = gdf["POP_EST"]

    elif "pop_est" not in gdf.columns:
        np.random.seed(0)
        gdf["pop_est"] = np.random.randint(
            1_000_000,
            1_000_000_000,
            size=len(gdf)
        )

    gdf["pop_est"] = gdf["pop_est"].fillna(0)

    return gdf, country_col


# Load dataset
try:
    if not DATA_PATH.exists():
        st.error(
            "Shapefile not found! Please check that the "
            "ne_110m_admin_0_countries folder is in the same "
            "directory as app.py."
        )
        st.stop()

    gdf, country_col = load_data()

except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()


# ==========================================
# 3. CALCULATE POPULATION DENSITY
# ==========================================

# Convert to equal-area projection to calculate area
gdf_area = gdf.to_crs(epsg=6933)

# Convert area from square metres to square kilometres
area_km2 = gdf_area.geometry.area / 1_000_000

# Calculate population density
gdf["pop_density"] = np.where(
    area_km2 > 0,
    gdf["pop_est"] / area_km2,
    0
)


# ==========================================
# 4. SIDEBAR MAP SELECTION
# ==========================================

st.sidebar.header("Choose a Map Type")

map_type = st.sidebar.radio(
    "Select Map Type",
    [
        "Map Cosmetics",
        "Map Visual Hierarchy",
        "Choropleth Map",
        "Geo Projection"
    ]
)


# ==========================================
# 5. MAP COSMETICS EXAMPLE
# ==========================================

if map_type == "Map Cosmetics":

    st.header("1. Map Cosmetics")

    st.write(
        "Customize the appearance of the map using "
        "different colours and backgrounds."
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    gdf.plot(
        ax=ax,
        color="skyblue",
        edgecolor="black",
        linewidth=0.5
    )

    ax.set_title(
        "Styled Map Example",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_facecolor("lightgrey")
    ax.set_axis_off()

    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# 6. MAP VISUAL HIERARCHY EXAMPLE
# ==========================================

elif map_type == "Map Visual Hierarchy":

    st.header("2. Map Visual Hierarchy")

    st.write(
        "Highlight selected countries in red while "
        "keeping the remaining countries in sky blue."
    )

    # Default highlighted countries
    default_countries = [
        "Brazil",
        "India",
        "China",
        "Australia",
        "South Africa",
        "Russia"
    ]

    # Keep only countries available in the dataset
    available_countries = gdf[country_col].dropna().unique()

    default_countries = [
        country for country in default_countries
        if country in available_countries
    ]

    # Allow users to select countries
    highlighted_countries = st.multiselect(
        "Select countries to highlight in red",
        sorted(available_countries),
        default=default_countries
    )

    # Assign colours
    gdf["color"] = gdf[country_col].apply(
        lambda x: "red"
        if x in highlighted_countries
        else "skyblue"
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    gdf.plot(
        color=gdf["color"],
        edgecolor="black",
        linewidth=0.5,
        ax=ax
    )

    ax.set_title(
        "Map Visual Hierarchy with Selected Countries Highlighted in Red",
        fontsize=13
    )

    ax.set_axis_off()

    st.pyplot(fig)
    plt.close(fig)

    st.write("Highlighted Countries:")

    if highlighted_countries:
        st.write(", ".join(highlighted_countries))
    else:
        st.write("No countries selected.")


# ==========================================
# 7. CHOROPLETH MAP EXAMPLE
# ==========================================

elif map_type == "Choropleth Map":

    st.header("3. Choropleth Map")

    st.write(
        "Create a choropleth map to visualize population "
        "density across different countries."
    )

    # Select colour scheme
    color_choice = st.selectbox(
        "Choose Color Scheme",
        ["OrRd", "YlGn", "Blues", "Purples", "Greens"],
        index=0
    )

    # Log transformation checkbox
    use_log_scale = st.checkbox(
        "Use Log Transformation",
        value=True
    )

    # Create a copy of the dataset
    plot_data = gdf.copy()

    # Apply log transformation
    if use_log_scale:
        plot_data["display_density"] = np.log1p(
            plot_data["pop_density"]
        )
    else:
        plot_data["display_density"] = plot_data["pop_density"]

    # Plot choropleth map
    fig, ax = plt.subplots(figsize=(12, 7))

    plot_data.plot(
        column="display_density",
        cmap=color_choice,
        edgecolor="black",
        linewidth=0.5,
        legend=True,
        ax=ax,
        missing_kwds={
            "color": "lightgrey",
            "label": "Missing data"
        }
    )

    ax.set_title(
        f"Population Density by Country (Color Scheme: {color_choice})",
        fontsize=13
    )

    ax.set_axis_off()

    st.pyplot(fig)
    plt.close(fig)

    st.write("Selected Color Scheme:", color_choice)

    if use_log_scale:
        st.write("Log Transformation: Enabled")
    else:
        st.write("Log Transformation: Disabled")


# ==========================================
# 8. GEO PROJECTION EXAMPLE
# ==========================================

elif map_type == "Geo Projection":

    st.header("4. Geo Projection")

    st.write(
        "Explore different geographical projections and "
        "observe how they affect the appearance of the map."
    )

    # Available map projections
    projection_options = {
        "WGS 84 (Plate Carree)": "EPSG:4326",
        "Mercator": "EPSG:3395",
        "Mollweide": "+proj=moll +datum=WGS84 +units=m +no_defs",
        "Robinson": "+proj=robin +datum=WGS84 +units=m +no_defs",
        "Equal Earth": "EPSG:8857"
    }

    # Select projection
    proj_choice = st.selectbox(
        "Select a Projection",
        list(projection_options.keys())
    )

    # Select colour scheme
    color_choice = st.selectbox(
        "Choose Color Scheme",
        ["OrRd", "YlGn", "Blues", "Purples", "Greens"],
        index=0
    )

    # Log transformation checkbox
    use_log_scale = st.checkbox(
        "Use Log Transformation for Population Density",
        value=True
    )

    # Prepare dataset
    plot_data = gdf.copy()

    if use_log_scale:
        plot_data["display_density"] = np.log1p(
            plot_data["pop_density"]
        )
    else:
        plot_data["display_density"] = plot_data["pop_density"]

    # Convert to selected projection
    gdf_proj = plot_data.to_crs(
        projection_options[proj_choice]
    )

    # Plot projected map
    fig, ax = plt.subplots(figsize=(12, 7))

    gdf_proj.plot(
        column="display_density",
        cmap=color_choice,
        edgecolor="black",
        linewidth=0.5,
        legend=True,
        ax=ax,
        missing_kwds={
            "color": "lightgrey",
            "label": "Missing data"
        }
    )

    ax.set_title(
        f"Map in {proj_choice} Projection",
        fontsize=14
    )

    ax.set_axis_off()

    st.pyplot(fig)
    plt.close(fig)

    st.write("Selected Projection:", proj_choice)
    st.write("Selected Color Scheme:", color_choice)

    if use_log_scale:
        st.write("Log Transformation: Enabled")
    else:
        st.write("Log Transformation: Disabled")


# ==========================================
# 9. SIDEBAR FOOTER
# ==========================================

st.sidebar.markdown("---")
st.sidebar.caption("TFB3133/TEB3133 Data Visualization")
st.sidebar.caption("Lab 7 - September 2026")
