import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

# Set page configuration
st.set_page_config(page_title="Soil Sense", page_icon="🌱", layout="wide")

# Custom CSS for green theme
st.markdown("""
<style>
    .main-header {
        color: #2E7D32;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        color: #2E7D32;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stSelectbox > div > div {
        border-color: #4CAF50;
    }
    .metric-card {
        background-color: #E8F5E8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
    }
</style>
""",
            unsafe_allow_html=True)

# Logo and Title
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("attached_assets/soilsense_logo.png", width=300)

st.markdown('<h1 class="main-header">Soil Sense</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">🌱 Agricultural Sensor Data Monitoring Platform</p>',
    unsafe_allow_html=True)

# Data source section
st.markdown("---")
st.markdown('<h3 style="color: #2E7D32;">📊 Data Source Selection</h3>',
            unsafe_allow_html=True)


# Get available experiment files
def get_available_experiments():
    """Get list of available experiment files"""
    experiments = []

    # Check for available TXT files in attached_assets
    import os

    # Try different possible paths (local vs cloud environment)

    possible_paths = [
        "./attached_assets",  # Cloud/Git environment
        "attached_assets",  # Alternative cloud path
        #"C:/Users/USER001/Desktop/SoilSense/attached_assets",  # Your local path
    ]

    assets_path = None
    for path in possible_paths:
        if os.path.exists(path):
            assets_path = path
            break

    if assets_path:
        # Look for Experiment files specifically (Experiment1 to Experiment7)
        for i in range(1, 8):  # Experiment1 through Experiment7
            filename_txt = f"Experiment{i}.txt"
            filename_TXT = f"Experiment{i}.TXT"

            file_path = None
            if os.path.exists(os.path.join(assets_path, filename_txt)):
                file_path = os.path.join(assets_path, filename_txt)
            elif os.path.exists(os.path.join(assets_path, filename_TXT)):
                file_path = os.path.join(assets_path, filename_TXT)

            if file_path:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read().strip()
                        # Check if file has content and contains timestamp data
                        if content and ('/' in content or ':' in content):
                            experiments.append(f"Experiment{i}")
                except:
                    continue

    # Sort from newest to oldest (7, 6, 5, 4, 3, 2, 1)
    experiments.sort(key=lambda x: int(x.replace('Experiment', '')),
                     reverse=True)

    return experiments


# Get available experiments
available_experiments = get_available_experiments()

if available_experiments:
    selected_experiment = st.selectbox("Choose data source:",
                                       available_experiments,
                                       index=0)
else:
    st.error(
        "No experiment files found. Please ensure Experiment1.txt through Experiment7.txt are in the attached_assets folder."
    )
    selected_experiment = None

uploaded_file = None


# Function to parse experiment TXT files with correct column structure
@st.cache_data
def parse_experiment_file(file_path):
    """Parse experiment TXT file with correct column structure for your data"""
    try:
        with open(file_path, 'r') as file:
            content = file.read()

        lines = content.strip().split('\n')

        # Skip the header line and parse data
        data_rows = []
        for line in lines:  # Process all lines (no header to skip)
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            parts = line.split(',')  # Use comma separator
            if len(
                    parts
            ) >= 20:  # Ensure we have enough columns based on your data structure
                try:
                    # Parse timestamp from first column
                    time_str = parts[0].strip()
                    if time_str and '/' in time_str:
                        # Handle different time formats
                        try:
                            if ':' in time_str:
                                timestamp = datetime.strptime(
                                    time_str, '%H:%M:%S %d/%m/%Y')
                            else:
                                timestamp = datetime.strptime(
                                    time_str, '%d/%m/%Y')
                        except:
                            continue

                        # Parse sensor values - convert to float, handle empty values
                        values = []
                        for val in parts[1:]:
                            try:
                                values.append(
                                    float(val.strip()) if val.strip() else 0.0)
                            except:
                                values.append(0.0)

                        data_rows.append([timestamp] + values)
                except:
                    continue  # Skip problematic lines

        # Define column names based on your actual experiment data structure
        columns = [
            'timestamp', 'CO2SCD30A [ppm]', 'Temperature_SCD30A [°C]',
            'RHSCD30A [%]', 'CO2SCD30B [ppm]', 'Temperature_SCD30B [°C]',
            'RHSCD30B [%]', 'CO2SCD30C [ppm]', 'Temperature_SCD30C [°C]',
            'RHSCD30C [%]', 'CO2SCD30D [ppm]', 'Temperature_SCD30D [°C]',
            'RHSCD30D [%]', 'oxygenDa_A [%Vol]', 'oxygenDa_B [%Vol]',
            'oxygenDa_C [%Vol]', 'oxygenDa_D [%Vol]',
            'oxygenBo_airTemp_A [°C]', 'oxygenBo_airTemp_B [°C]',
            'oxygenBo_airTemp_C [°C]', 'oxygenBo_airTemp_D [°C]',
            'measuredvbat [V]'
        ]

        # Truncate columns to match available data
        max_cols = min(len(columns), len(data_rows[0]) if data_rows else 0)
        columns = columns[:max_cols]

        df = pd.DataFrame(data_rows, columns=columns)

        # Filter out rows with all zero sensor values (inactive periods)
        if len(df) > 0:
            df = df[df.iloc[:, 1:].sum(axis=1)
                    > 0]  # Remove rows where all sensor values are 0

        return df

    except Exception as e:
        st.error(f"Error parsing file {file_path}: {e}")
        return pd.DataFrame()


# Cache the data generation to improve performance
@st.cache_data
def load_experiment_data(experiment_name):
    """Load data from selected experiment file"""
    if not experiment_name:
        return pd.DataFrame()

    # Try different possible paths (local vs cloud environment)
    possible_paths = [
        "attached_assets",  # Cloud environment
        "C:/Users/USER001/Desktop/SoilSense/attached_assets",  # Your local path
        "./attached_assets"  # Relative path
    ]

    # Try each path with both .txt and .TXT extensions
    for base_path in possible_paths:
        if os.path.exists(base_path):
            # Try .txt extension
            file_path_txt = os.path.join(base_path, f"{experiment_name}.txt")
            if os.path.exists(file_path_txt):
                return parse_experiment_file(file_path_txt)

            # Try .TXT extension
            file_path_TXT = os.path.join(base_path, f"{experiment_name}.TXT")
            if os.path.exists(file_path_TXT):
                return parse_experiment_file(file_path_TXT)

    return pd.DataFrame()


# Function to create separate plots by sensor type
def create_sensor_plots(df):
    """Create separate plots for CO2, Temperature, and Humidity sensors"""
    if df.empty:
        return []

    # Group sensors by type
    co2_sensors = [col for col in df.columns if col.startswith('CO2')]
    temp_sensors = [col for col in df.columns if col.startswith('Temperature')]
    humidity_sensors = [col for col in df.columns if col.startswith('RH')]
    oxygen_sensors = [col for col in df.columns if col.startswith('oxygenDa')]
    oxygen_temp_sensors = [
        col for col in df.columns if col.startswith('oxygenBo_airTemp')
    ]

    plots = []

    # CO2 Plot
    if co2_sensors:
        fig_co2 = go.Figure()
        for sensor in co2_sensors:
            fig_co2.add_trace(
                go.Scatter(x=df['timestamp'],
                           y=df[sensor],
                           mode='lines',
                           name=sensor.replace('_', ' '),
                           line=dict(width=2)))
        fig_co2.update_layout(title="🌱 CO2 Levels (ppm)",
                              xaxis_title="Time",
                              yaxis_title="CO2 (ppm)",
                              hovermode='x unified',
                              height=400)
        plots.append(("CO2 Sensors", fig_co2))

    # Temperature Plot
    if temp_sensors:
        fig_temp = go.Figure()
        for sensor in temp_sensors:
            fig_temp.add_trace(
                go.Scatter(x=df['timestamp'],
                           y=df[sensor],
                           mode='lines',
                           name=sensor.replace('_', ' '),
                           line=dict(width=2)))
        fig_temp.update_layout(title="🌡️ Temperature Readings (°C)",
                               xaxis_title="Time",
                               yaxis_title="Temperature (°C)",
                               hovermode='x unified',
                               height=400)
        plots.append(("Temperature Sensors", fig_temp))

    # Humidity Plot
    if humidity_sensors:
        fig_humidity = go.Figure()
        for sensor in humidity_sensors:
            fig_humidity.add_trace(
                go.Scatter(x=df['timestamp'],
                           y=df[sensor],
                           mode='lines',
                           name=sensor.replace('_', ' '),
                           line=dict(width=2)))
        fig_humidity.update_layout(title="💧 Humidity Levels (%)",
                                   xaxis_title="Time",
                                   yaxis_title="Humidity (%)",
                                   hovermode='x unified',
                                   height=400)
        plots.append(("Humidity Sensors", fig_humidity))

    # Oxygen Plot
    if oxygen_sensors:
        fig_oxygen = go.Figure()
        for sensor in oxygen_sensors:
            fig_oxygen.add_trace(
                go.Scatter(x=df['timestamp'],
                           y=df[sensor],
                           mode='lines',
                           name=sensor.replace('_', ' '),
                           line=dict(width=2)))
        fig_oxygen.update_layout(title="🫁 Oxygen Levels (%Vol)",
                                 xaxis_title="Time",
                                 yaxis_title="Oxygen (%Vol)",
                                 hovermode='x unified',
                                 height=400)
        plots.append(("Oxygen Sensors", fig_oxygen))

    # Oxygen Temperature Plot
    if oxygen_temp_sensors:
        fig_oxygen_temp = go.Figure()
        for sensor in oxygen_temp_sensors:
            fig_oxygen_temp.add_trace(
                go.Scatter(x=df['timestamp'],
                           y=df[sensor],
                           mode='lines',
                           name=sensor.replace('_', ' '),
                           line=dict(width=2)))
        fig_oxygen_temp.update_layout(
            title="🌡️ Oxygen Sensor Temperature (°C)",
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
            hovermode='x unified',
            height=400)
        plots.append(("Oxygen Temperature", fig_oxygen_temp))

    return plots


# Create correlation heatmap
def create_correlation_heatmap(df, columns):
    """Create a correlation heatmap for selected sensors"""
    if len(columns) < 2:
        return None

    corr_df = df[columns].corr()

    fig = px.imshow(corr_df,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1,
                    title="Sensor Correlation Heatmap")

    fig.update_layout(height=400)
    return fig


# Alert System Functions
def check_sensor_thresholds(df, thresholds):
    """Check current sensor readings against defined thresholds"""
    if df.empty or len(df) == 0:
        return []

    alerts = []
    latest_row = df.iloc[-1]
    current_time = datetime.now()

    # Check CO2 sensors
    co2_columns = [col for col in df.columns if col.startswith('CO2')]
    for col in co2_columns:
        value = latest_row[col]
        if value < thresholds['co2']['min']:
            alerts.append({
                'sensor': col,
                'message': f'CO2 level too low',
                'current_value': value,
                'threshold': thresholds['co2']['min'],
                'severity': 'medium',
                'timestamp': current_time
            })
        elif value > thresholds['co2']['max']:
            alerts.append({
                'sensor': col,
                'message': f'CO2 level too high',
                'current_value': value,
                'threshold': thresholds['co2']['max'],
                'severity': 'high',
                'timestamp': current_time
            })

    # Check Temperature sensors
    temp_columns = [col for col in df.columns if col.startswith('Temperature')]
    for col in temp_columns:
        value = latest_row[col]
        if value < thresholds['temperature']['min']:
            alerts.append({
                'sensor': col,
                'message': f'Temperature too low',
                'current_value': value,
                'threshold': thresholds['temperature']['min'],
                'severity': 'high',
                'timestamp': current_time
            })
        elif value > thresholds['temperature']['max']:
            alerts.append({
                'sensor': col,
                'message': f'Temperature too high',
                'current_value': value,
                'threshold': thresholds['temperature']['max'],
                'severity': 'high',
                'timestamp': current_time
            })

    # Check Humidity sensors
    humidity_columns = [col for col in df.columns if col.startswith('RH')]
    for col in humidity_columns:
        value = latest_row[col]
        if value < thresholds['humidity']['min']:
            alerts.append({
                'sensor': col,
                'message': f'Humidity too low',
                'current_value': value,
                'threshold': thresholds['humidity']['min'],
                'severity': 'medium',
                'timestamp': current_time
            })
        elif value > thresholds['humidity']['max']:
            alerts.append({
                'sensor': col,
                'message': f'Humidity too high',
                'current_value': value,
                'threshold': thresholds['humidity']['max'],
                'severity': 'medium',
                'timestamp': current_time
            })

    # Check Battery voltage
    if 'Battery_Volt' in df.columns:
        value = latest_row['Battery_Volt']
        if value < thresholds['battery']['min']:
            alerts.append({
                'sensor': 'Battery_Volt',
                'message': f'Battery voltage low',
                'current_value': value,
                'threshold': thresholds['battery']['min'],
                'severity': 'high',
                'timestamp': current_time
            })

    return alerts


# Load data based on user selection
df = load_experiment_data(
    selected_experiment) if selected_experiment else pd.DataFrame()

if not df.empty:
    # Display data source info
    st.success(f"✅ Data loaded from: {selected_experiment}")

    # Main application interface
    st.markdown("---")
    st.markdown('<h3 style="color: #2E7D32;">📈 Soil Sense Data Analysis</h3>',
                unsafe_allow_html=True)

    # Display data info
    st.write(f"📊 **Records**: {len(df)}")
    if 'timestamp' in df.columns and len(df) > 0:
        st.write(
            f"⏰ **Time Range**: {df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}"
        )

    # Get sensor columns (all except timestamp)
    sensor_columns = [col for col in df.columns if col != 'timestamp']

    # Create 4 tabs for different visualizations
    viz_tab1, viz_tab2, viz_tab3, alert_tab = st.tabs(
        ["📈 Time Series", "📊 Statistics", "🔄 Correlations", "🚨 Alert System"])

    with viz_tab1:
        # Time range selection
        st.subheader("Time Range Selection")

        # Get min and max timestamps
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()

        # Create two columns for start and end time
        col1, col2 = st.columns(2)

        with col1:
            start_time = st.date_input("Start Date",
                                       value=min_time.date(),
                                       min_value=min_time.date(),
                                       max_value=max_time.date())
            start_hour = st.time_input("Start Time", value=min_time.time())

        with col2:
            end_time = st.date_input("End Date",
                                     value=max_time.date(),
                                     min_value=min_time.date(),
                                     max_value=max_time.date())
            end_hour = st.time_input("End Time", value=max_time.time())

        # Combine date and time inputs
        start_datetime = pd.to_datetime(f"{start_time} {start_hour}")
        end_datetime = pd.to_datetime(f"{end_time} {end_hour}")

        # Filter dataframe based on selected time range
        if start_datetime <= end_datetime:
            filtered_df = df[(df['timestamp'] >= start_datetime)
                             & (df['timestamp'] <= end_datetime)]

            if not filtered_df.empty:
                # Create separate plots by sensor type
                plots = create_sensor_plots(filtered_df)

                # Display each plot type
                for plot_title, fig in plots:
                    st.subheader(plot_title)
                    st.plotly_chart(fig, use_container_width=True)

                # Show data summary for filtered range
                st.write(
                    f"📊 Showing {len(filtered_df)} data points from {start_time} to {end_time}"
                )

                # Raw Data Table section (only in Time Series tab)
                st.markdown("---")
                with st.expander("View Raw Data", expanded=False):
                    st.subheader("Raw Data (Filtered by Time Range)")
                    st.dataframe(filtered_df.head(100),
                                 use_container_width=True)

                    # Option to download filtered data
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download Filtered Dataset",
                        data=csv,
                        file_name=
                        f"soil_sense_data_{start_time}_to_{end_time}.csv",
                        mime="text/csv")
            else:
                st.warning("No data found in the selected time range")
        else:
            st.error("Start time must be before end time")

    with viz_tab2:
        st.subheader("Sensor Statistics")

        # Create a dataframe with statistics for all sensors
        stats = []
        for col in sensor_columns:
            stats.append({
                "Sensor": col,
                "Min": f"{df[col].min():.1f}",
                "Max": f"{df[col].max():.1f}",
                "Average": f"{df[col].mean():.1f}",
                "Std Dev": f"{df[col].std():.1f}"
            })

        # Display statistics as a table
        st.table(pd.DataFrame(stats))

        # Show the latest readings
        st.subheader("Latest Readings")
        latest = df.iloc[-1:].copy()
        latest_time = latest['timestamp'].iloc[0]
        st.write(f"Time: {latest_time}")

        # Create metrics in two rows
        sensors_per_row = max(1, len(sensor_columns) // 2)

        # First row
        if len(sensor_columns) > 0:
            first_row_sensors = sensor_columns[:sensors_per_row]
            cols1 = st.columns(len(first_row_sensors))
            for i, sensor in enumerate(first_row_sensors):
                if len(df) >= 2:
                    prev_value = df.iloc[-2][sensor]
                    current_value = latest[sensor].iloc[0]
                    delta = current_value - prev_value
                else:
                    current_value = latest[sensor].iloc[0]
                    delta = 0

                with cols1[i]:
                    st.markdown(f"""
                    <div style='border: 1px solid #e0e0e0; padding: 8px; border-radius: 4px; text-align: center;'>
                        <p style='font-size: 10px; margin: 0; color: #666;'>{sensor}</p>
                        <p style='font-size: 18px; margin: 2px 0; font-weight: bold;'>{current_value:.1f}</p>
                        <p style='font-size: 12px; margin: 0; color: {"green" if delta >= 0 else "red"};'>{"↑" if delta > 0 else "↓" if delta < 0 else "→"} {delta:.1f}</p>
                    </div>
                    """,
                                unsafe_allow_html=True)

        # Second row
        if len(sensor_columns) > sensors_per_row:
            second_row_sensors = sensor_columns[sensors_per_row:]
            cols2 = st.columns(len(second_row_sensors))
            for i, sensor in enumerate(second_row_sensors):
                if len(df) >= 2:
                    prev_value = df.iloc[-2][sensor]
                    current_value = latest[sensor].iloc[0]
                    delta = current_value - prev_value
                else:
                    current_value = latest[sensor].iloc[0]
                    delta = 0

                with cols2[i]:
                    st.markdown(f"""
                    <div style='border: 1px solid #e0e0e0; padding: 8px; border-radius: 4px; text-align: center;'>
                        <p style='font-size: 10px; margin: 0; color: #666;'>{sensor}</p>
                        <p style='font-size: 18px; margin: 2px 0; font-weight: bold;'>{current_value:.1f}</p>
                        <p style='font-size: 12px; margin: 0; color: {"green" if delta >= 0 else "red"};'>{"↑" if delta > 0 else "↓" if delta < 0 else "→"} {delta:.1f}</p>
                    </div>
                    """,
                                unsafe_allow_html=True)

        # CO2 Regression Analysis
        st.subheader("CO2 Regression Analysis")

        # Get CO2 sensors
        co2_sensors = [col for col in sensor_columns if 'CO2' in col]

        if co2_sensors:
            # Select CO2 sensor for regression
            selected_co2_sensor = st.selectbox(
                "Select CO2 sensor for regression analysis:", co2_sensors)

            # Select regression type
            regression_type = st.selectbox(
                "Select regression type:",
                ["Linear", "Exponential", "Logarithmic"])

            if selected_co2_sensor and len(df) >= 10:
                # Prepare data for regression
                df_reg = df[['timestamp', selected_co2_sensor]].copy()
                df_reg = df_reg.dropna()

                # Convert timestamps to numeric values (hours from start)
                df_reg['hours'] = (
                                          df_reg['timestamp'] -
                                          df_reg['timestamp'].min()).dt.total_seconds() / 3600

                x = df_reg['hours'].values
                y = df_reg[selected_co2_sensor].values

                # Apply CO2 limit constraint (max 40,000 ppm)
                y = np.minimum(y, 40000)

                # Prediction settings
                st.write("**Prediction Settings**")
                prediction_hours = st.number_input("Predict ahead (hours)", min_value=1, max_value=168, value=24)

                # Perform regression based on selected type
                try:
                    from sklearn.preprocessing import PolynomialFeatures
                    from sklearn.linear_model import LinearRegression
                    from sklearn.pipeline import Pipeline
                    from scipy import stats
                    from sklearn.metrics import r2_score
                    import numpy as np
                    import pandas as pd

                    if regression_type == "Linear":
                        # Linear regression
                        model = LinearRegression()
                        X = x.reshape(-1, 1)
                        model.fit(X, y)
                        y_pred = model.predict(X)
                        y_pred = np.minimum(y_pred, 40000)  # Apply constraint

                        # Calculate statistics
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                        equation = f"CO₂ = {slope:.4f}t + {intercept:.2f}"

                        # Debug: Check if p_value is valid
                        if np.isnan(p_value) or p_value <= 0:
                            # Alternative calculation using t-test
                            n = len(x)
                            if n > 2:
                                t_stat = slope / (std_err + 1e-10)  # Avoid division by zero
                                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                            else:
                                p_value = 1.0

                        # Future prediction
                        future_x = x[-1] + prediction_hours
                        future_pred = model.predict([[future_x]])[0]

                    elif regression_type == "Exponential":
                        # Exponential regression (log transform)
                        y_positive = np.maximum(y, 1)  # Ensure positive values
                        log_y = np.log(y_positive)
                        model = LinearRegression()
                        X = x.reshape(-1, 1)
                        model.fit(X, log_y)
                        log_y_pred = model.predict(X)
                        y_pred = np.exp(log_y_pred)
                        y_pred = np.minimum(y_pred, 40000)  # Apply constraint

                        # Calculate statistics for exponential fit
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_y)
                        equation = f"CO₂ = {np.exp(intercept):.4f} * e^({slope:.4f}t)"

                        # Future prediction
                        future_x = x[-1] + prediction_hours
                        future_log_pred = model.predict([[future_x]])[0]
                        future_pred = np.exp(future_log_pred)

                    elif regression_type == "Logarithmic":
                        # Logarithmic regression
                        x_positive = np.maximum(
                            x, 0.1)  # Ensure positive x values
                        log_x = np.log(x_positive + 1)  # Add 1 to handle x=0
                        model = LinearRegression()
                        X = log_x.reshape(-1, 1)
                        model.fit(X, y)
                        y_pred = model.predict(X)
                        y_pred = np.minimum(y_pred, 40000)  # Apply constraint

                        # Calculate statistics for logarithmic fit
                        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, y)
                        equation = f"CO₂ = {slope:.4f} * ln(t+1) + {intercept:.2f}"

                        # Future prediction
                        future_x = x[-1] + prediction_hours
                        future_log_x = np.log(max(future_x, 0.1) + 1)
                        future_pred = model.predict([[future_log_x]])[0]

                    # Apply constraint to future prediction
                    future_pred = min(future_pred, 40000)

                    # Calculate R-squared
                    r2 = r2_score(y, y_pred)

                    # Create regression plot
                    fig_reg = go.Figure()

                    # Original data
                    fig_reg.add_trace(
                        go.Scatter(x=df_reg['timestamp'],
                                   y=y,
                                   mode='markers',
                                   name='Actual Data',
                                   marker=dict(color='blue', size=4)))

                    # Regression line
                    fig_reg.add_trace(
                        go.Scatter(x=df_reg['timestamp'],
                                   y=y_pred,
                                   mode='lines',
                                   name=f'{regression_type} Regression',
                                   line=dict(color='red', width=2)))

                    # Add prediction point
                    future_time = df_reg['timestamp'].iloc[-1] + pd.Timedelta(hours=prediction_hours)
                    fig_reg.add_trace(
                        go.Scatter(x=[future_time],
                                   y=[future_pred],
                                   mode='markers',
                                   name=f'Prediction ({prediction_hours}h)',
                                   marker=dict(color='orange', size=12, symbol='star'))
                    )

                    # Add 40,000 ppm limit line
                    fig_reg.add_hline(y=40000,
                                      line_dash="dash",
                                      line_color="orange",
                                      annotation_text="CO2 Limit (40,000 ppm)")

                    fig_reg.update_layout(
                        title=
                        f"{regression_type} Regression Analysis for {selected_co2_sensor}",
                        xaxis_title="Time",
                        yaxis_title="CO2 (ppm)",
                        hovermode='x unified',
                        height=400)

                    st.plotly_chart(fig_reg, use_container_width=True)

                    # Display regression statistics
                    col1, col2 = st.columns(2)

                    with col1:
                        # Format p-value properly
                        if p_value < 0.001:
                            p_value_str = f"{p_value:.2e}"
                        else:
                            p_value_str = f"{p_value:.6f}"

                        st.info(f"""
                        **Regression Statistics:**
                        - Type: {regression_type}
                        - R-squared: {r2:.4f}
                        - P-value: {p_value_str}
                        - Data points: {len(y)}
                        """)

                    with col2:
                        st.info(f"""
                        **Equation:**
                        {equation}

                        **Prediction:**
                        In {prediction_hours} hours: {future_pred:.1f} ppm
                        """)

                    # Interpretation of p-value
                    if p_value < 0.001:
                        p_interpretation = "Highly significant (p < 0.001)"
                    elif p_value < 0.01:
                        p_interpretation = "Very significant (p < 0.01)"
                    elif p_value < 0.05:
                        p_interpretation = "Significant (p < 0.05)"
                    else:
                        p_interpretation = "Not statistically significant (p ≥ 0.05)"

                    st.write(f"**Statistical Significance:** {p_interpretation}")

                    if future_pred >= 40000:
                        st.warning("⚠️ Predicted CO2 level reaches the maximum constraint (40,000 ppm)")

                    # Model Comparison Section
                    st.subheader("📊 Model Comparison")

                    # Run all three models for comparison
                    comparison_results = {}

                    for model_type in ["Linear", "Exponential", "Logarithmic"]:
                        try:
                            if model_type == "Linear":
                                temp_model = LinearRegression()
                                temp_X = x.reshape(-1, 1)
                                temp_model.fit(temp_X, y)
                                temp_y_pred = temp_model.predict(temp_X)
                                temp_slope, temp_intercept, temp_r_value, temp_p_value, temp_std_err = stats.linregress(
                                    x, y)

                            elif model_type == "Exponential":
                                temp_y_positive = np.maximum(y, 1)
                                temp_log_y = np.log(temp_y_positive)
                                temp_model = LinearRegression()
                                temp_X = x.reshape(-1, 1)
                                temp_model.fit(temp_X, temp_log_y)
                                temp_log_y_pred = temp_model.predict(temp_X)
                                temp_y_pred = np.exp(temp_log_y_pred)
                                temp_slope, temp_intercept, temp_r_value, temp_p_value, temp_std_err = stats.linregress(
                                    x, temp_log_y)

                            elif model_type == "Logarithmic":
                                temp_x_positive = np.maximum(x, 0.1)
                                temp_log_x = np.log(temp_x_positive + 1)
                                temp_model = LinearRegression()
                                temp_X = temp_log_x.reshape(-1, 1)
                                temp_model.fit(temp_X, y)
                                temp_y_pred = temp_model.predict(temp_X)
                                temp_slope, temp_intercept, temp_r_value, temp_p_value, temp_std_err = stats.linregress(
                                    temp_log_x, y)

                            temp_y_pred = np.minimum(temp_y_pred, 40000)
                            temp_r2 = r2_score(y, temp_y_pred)

                            # Fix p-value if needed
                            if np.isnan(temp_p_value) or temp_p_value <= 0:
                                n = len(x)
                                if n > 2:
                                    t_stat = temp_slope / (temp_std_err + 1e-10)
                                    temp_p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                                else:
                                    temp_p_value = 1.0

                            comparison_results[model_type] = {
                                'r2': temp_r2,
                                'p_value': temp_p_value,
                                'slope': temp_slope,
                                'intercept': temp_intercept
                            }
                        except:
                            comparison_results[model_type] = {
                                'r2': 0.0,
                                'p_value': 1.0,
                                'slope': 0.0,
                                'intercept': 0.0
                            }

                    # Find best model based on R²
                    best_model = max(comparison_results.keys(), key=lambda k: comparison_results[k]['r2'])
                    best_r2 = comparison_results[best_model]['r2']

                    # Generate comparison text
                    significance_count = sum(1 for result in comparison_results.values() if result['p_value'] < 0.05)
                    highly_significant_count = sum(
                        1 for result in comparison_results.values() if result['p_value'] < 0.001)

                    if highly_significant_count == 3:
                        significance_text = "All three models showed a statistically significant relationship (p < 0.001)"
                    elif significance_count == 3:
                        significance_text = "All three models showed a statistically significant relationship (p < 0.05)"
                    elif significance_count == 2:
                        significant_models = [model for model, result in comparison_results.items() if
                                              result['p_value'] < 0.05]
                        significance_text = f"Two models ({', '.join(significant_models)}) showed statistically significant relationships (p < 0.05)"
                    elif significance_count == 1:
                        significant_model = \
                        [model for model, result in comparison_results.items() if result['p_value'] < 0.05][0]
                        significance_text = f"Only the {significant_model} model showed a statistically significant relationship (p < 0.05)"
                    else:
                        significance_text = "None of the models showed statistically significant relationships (p ≥ 0.05)"

                    comparison_text = f"{significance_text}, but the level of fit varied between them. The {best_model.lower()} model provided the best fit to the data (R² = {best_r2:.4f})."

                    st.info(f"**Model Comparison Summary:**\n{comparison_text}")

                    # Detailed comparison table
                    comparison_df = pd.DataFrame.from_dict(comparison_results, orient='index')
                    comparison_df['Model'] = comparison_df.index
                    comparison_df = comparison_df[['Model', 'r2', 'p_value']]
                    comparison_df.columns = ['Model Type', 'R² Score', 'P-value']
                    comparison_df['R² Score'] = comparison_df['R² Score'].round(4)
                    comparison_df['P-value'] = comparison_df['P-value'].apply(
                        lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.6f}")

                    st.dataframe(comparison_df, use_container_width=True)

                except ImportError:
                    st.error(
                        "Regression analysis requires scikit-learn. Please install it to use this feature."
                    )
                except Exception as e:
                    st.error(f"Error performing regression analysis: {str(e)}")

            elif selected_co2_sensor and len(df) < 10:
                st.warning(
                    "Need at least 10 data points for regression analysis")
        else:
            st.info("No CO2 sensors found in the dataset")

    with viz_tab3:
        st.subheader("Sensor Correlations")

        # Comprehensive explanation of correlation analysis
        st.info("""
        **Correlation Analysis Explanation:**

        The correlation analysis is calculated based on the Pearson correlation coefficient between the selected sensor readings. This measures the linear relationship between different sensors over time.

        **Correlation values range from -1 to +1:**
        - **+1.0**: Perfect positive correlation (as one sensor increases, the other increases proportionally)
        - **+0.7 to +0.9**: Strong positive correlation
        - **+0.3 to +0.7**: Moderate positive correlation
        - **-0.3 to +0.3**: Weak or no linear relationship
        - **-0.3 to -0.7**: Moderate negative correlation
        - **-0.7 to -0.9**: Strong negative correlation
        - **-1.0**: Perfect negative correlation (as one sensor increases, the other decreases proportionally)
        """)

        # Select sensors for correlation
        correlation_sensors = st.multiselect(
            "Select sensors for correlation analysis:",
            sensor_columns,
            default=sensor_columns
            if len(sensor_columns) <= 4 else sensor_columns[:4])

        if len(correlation_sensors) >= 2:
            # Create correlation heatmap
            corr_fig = create_correlation_heatmap(df, correlation_sensors)
            st.plotly_chart(corr_fig, use_container_width=True)
        else:
            st.info("Please select at least two sensors to view correlations")

    with alert_tab:
        st.markdown('<h3 style="color: #2E7D32;">🚨 Smart Alert System</h3>',
                    unsafe_allow_html=True)
        st.write(
            "Set up automatic notifications when sensor readings exceed your defined thresholds"
        )

        # Alert configuration section
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📧 Email Notifications")
            email_enabled = st.checkbox("Enable email alerts",
                                        key="email_alerts")
            if email_enabled:
                user_email = st.text_input("Your email address:",
                                           placeholder="example@email.com")

        with col2:
            st.subheader("📱 SMS Notifications")
            sms_enabled = st.checkbox("Enable SMS alerts", key="sms_alerts")
            if sms_enabled:
                user_phone = st.text_input("Your phone number:",
                                           placeholder="+1234567890")

        # Threshold configuration
        st.subheader("⚙️ Alert Thresholds")

        # Create columns for different sensor types
        thresh_col1, thresh_col2, thresh_col3, thresh_col4 = st.columns(4)

        with thresh_col1:
            st.write("**CO2 Levels (ppm)**")
            co2_min = st.number_input("Min CO2:",
                                      value=400.0,
                                      step=50.0,
                                      key="co2_min")
            co2_max = st.number_input("Max CO2:",
                                      value=2000.0,
                                      step=50.0,
                                      key="co2_max")

        with thresh_col2:
            st.write("**Temperature (°C)**")
            temp_min = st.number_input("Min Temp:",
                                       value=15.0,
                                       step=1.0,
                                       key="temp_min")
            temp_max = st.number_input("Max Temp:",
                                       value=35.0,
                                       step=1.0,
                                       key="temp_max")

        with thresh_col3:
            st.write("**Humidity (%)**")
            hum_min = st.number_input("Min Humidity:",
                                      value=30.0,
                                      step=5.0,
                                      key="hum_min")
            hum_max = st.number_input("Max Humidity:",
                                      value=90.0,
                                      step=5.0,
                                      key="hum_max")

        with thresh_col4:
            st.write("**Battery (V)**")
            battery_min = st.number_input("Min Battery:",
                                          value=3.5,
                                          step=0.1,
                                          key="battery_min")

        # Check alerts button
        if st.button("🔍 Check Current Status", type="primary"):
            # Define thresholds based on user input
            thresholds = {
                'co2': {
                    'min': co2_min,
                    'max': co2_max
                },
                'temperature': {
                    'min': temp_min,
                    'max': temp_max
                },
                'humidity': {
                    'min': hum_min,
                    'max': hum_max
                },
                'battery': {
                    'min': battery_min
                }
            }

            # Check for alerts
            alerts = check_sensor_thresholds(df, thresholds)

            if alerts:
                st.error(f"🚨 {len(alerts)} Alert(s) Detected!")

                for alert in alerts:
                    severity_color = "🔴" if alert['severity'] == 'high' else "🟡"
                    st.warning(
                        f"{severity_color} **{alert['sensor']}**: {alert['message']} (Current: {alert['current_value']:.1f})"
                    )

                # Show notification status
                if email_enabled and 'user_email' in locals() and user_email:
                    st.success("📧 Email notification ready!")

                if sms_enabled and 'user_phone' in locals() and user_phone:
                    st.success("📱 SMS notification ready!")

            else:
                st.success("✅ All sensors are within normal ranges!")

else:
    st.error(
        "No data available. Please check your data source or upload a file.")
