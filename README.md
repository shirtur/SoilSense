# Soil Sense - Agricultural Sensor Data Monitoring Platform

A comprehensive Streamlit web application for monitoring and analyzing agricultural sensor data including CO2, temperature, humidity, and O2 levels from multiple experiments stored in Google Drive.

## Security

This README contains only placeholders for sensitive information. Do not include real credentials here. All secrets (SMTP credentials, Google Drive folder ID, etc.) must be stored securely:

Streamlit Cloud: Define secrets in App Settings , Secrets.

Local Development: Create a .streamlit/secrets.toml file (listed in .gitignore) with the same keys.

## Features
-Data Loading from Google Drive: Automatically download experiment .txt files from a public Google Drive folder using gdown.
-Multi-Experiment Support: Select and switch between multiple experiments; refresh experiments on demand.
-Real-time Data Visualization: Interactive time-series plots for CO2, temperature, humidity, O2, and sensor temperature.
-Dual-Axis Analysis: Compare CO2 vs O2 for any chamber on a dual-axis plot.
-Statistical Analysis: Regression (Linear, Exponential, Logarithmic) with prediction and model comparison; R^2, p-values, and detailed statistics.
-Correlation Heatmap: Pearson correlation heatmap for selected sensors with interactive tooltips.
-Data Filtering: Remove zeros, outliers (3 Standart Deviation, IQR, percentile), or sudden spikes with configurable handling (blank or average).
-Time-Range Selection: Filter data by date and time in the sidebar.
-Raw Data Export: View and download filtered data as CSV.
-Smart Alert System (Email): Configure thresholds for CO2, temperature, humidity, and battery; send email alerts via SMTP.
-Responsive Dashboard: Clean layout, custom green theme, and mobile compatibility.

## Installation

### Local Development

1. Clone the repository:

git clone https://github.com/shirtur/SoilSense.git
cd soil-sense


2. Install dependencies:

pip install -r requirements.txt


3. Configure email credentials in Streamlit's secrets management (e.g., via Streamlit Cloud App Settings , Secrets):

[email]
smtp_server = "smtp.gmail.com"
smtp_port = 587
username = "your_email@example.com"
password = "app_password"
from = "your_email@example.com"

4. Set your Google Drive folder ID (in main_app.py or via an environment variable):
FOLDER_ID = "YOUR_DRIVE_FOLDER_ID"

5. Run the application:
streamlit run app.py

### Cloud Deployment

#### Streamlit Cloud
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add your secrets in App Settings , Secrets
5. Set main_app.py as the main file and deploy.



## Configuration

### Environment Variables (Optional)

Google Drive Folder ID: YOUR_DRIVE_FOLDER_ID

Email Alerts: Set via Streamlit secrets (see above).

Thresholds & Filters: Configure via sidebar controls at runtime.


## Usage

1. **Refresh Experiments**: Click 'Refresh Experiments' in the sidebar to re-download files.
2. **Select Experiment**: Choose from available .txt experiments.
3. **Filter Data**: Apply data cleaning filters and select handling.
4. **Time Range**: Pick start/end date and time.
5. **Tabs**: 
Time Series: View sensor plots and dual-axis analysis.
Statistics: Summary table, latest readings, and CO2 regression.
Correlations: Select sensors for correlation heatmap.
Alerts: Configure and receive email notifications.
6. **Download**: Export filtered dataset as CSV.


## Data Format

Experiment .txt files should be comma-separated with columns:
timestamp, CO2SCD30A [ppm], Temperature_SCD30A [CelsiusDegree], RHSCD30A [%], ..., oxygenDa_A [%Vol], oxygenBo_airTemp_A [CelsiusDegree], ..., measuredvbat [V]
- `timestamp`: HH:MM:SS DD/MM/YYYY or DD/MM/YYYY
Sensor columns for chambers A-D:
    - `CO2SCD30*`: CO2 levels in parts per million [ppm]
    - `Temperature_SCD30*`: Temperature in Celsius [CelsiusDegree]
    - `RHSCD30*`: Humidity percentage [%]
    - `oxygenDa_*`: Oxygen percentage [%Vol]
    - `oxygenBo_airTemp_*`: Oxygen temperature [%Vol]
- `measuredvbat`: Battery voltage [V]


## Technologies Used

- **Frontend**: Streamlit
- **Data Loading**: gdown
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Statistics**: SciPy, Scikit-learn
- **Email**: smtplib via Streamlit secrets
- **Python Version**: Requires Python 3.8 or above

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions, please open a GitHub issue or contact the development team.