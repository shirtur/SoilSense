# Soil Sense - Agricultural Sensor Data Monitoring Platform

A comprehensive web application for monitoring and analyzing agricultural sensor data including CO2, temperature, humidity, and oxygen levels.

## Features

- **Real-time Data Visualization**: Interactive time-series plots for all sensor types
- **Statistical Analysis**: Comprehensive correlation analysis with detailed explanations
- **Alert System**: Configurable thresholds with email and SMS notifications
- **Responsive Dashboard**: Clean, professional interface with mobile compatibility
- **Multi-experiment Support**: Load and compare data from multiple experiments

## Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/soil-sense.git
cd soil-sense
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create Streamlit configuration:
```bash
mkdir .streamlit
cp streamlit_config.toml .streamlit/config.toml
```

4. Run the application:
```bash
streamlit run app.py
```

### Cloud Deployment

#### Streamlit Cloud
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with `app.py` as the main file

#### Heroku
1. Create a `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Deploy to Heroku:
```bash
heroku create your-app-name
git push heroku main
```

#### Railway
1. Connect your GitHub repository to [Railway](https://railway.app)
2. The app will auto-deploy using the provided configuration

## Configuration

### Environment Variables (Optional)

For email alerts:
- `SMTP_SERVER`: SMTP server address (e.g., smtp.gmail.com)
- `SMTP_PORT`: SMTP port (usually 587)
- `SMTP_USERNAME`: Your email address
- `SMTP_PASSWORD`: Your email password or app password

For SMS alerts:
- `TWILIO_ACCOUNT_SID`: Twilio Account SID
- `TWILIO_AUTH_TOKEN`: Twilio Auth Token
- `TWILIO_PHONE_NUMBER`: Twilio phone number

## Usage

1. **Select Experiment**: Choose from available experiments in the sidebar
2. **Dashboard**: View latest sensor readings and data summary
3. **Visualizations**: Explore interactive plots and correlation analysis
4. **Alerts**: Configure thresholds and notification settings

## Data Format

The application expects sensor data with these columns:
- `DateTime`: Timestamp for each reading
- `CO2_ppm`: CO2 levels in parts per million
- `Temperature_C`: Temperature in Celsius
- `Humidity_%`: Humidity percentage
- `Oxygen_%`: Oxygen percentage

## Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Statistics**: SciPy, Scikit-learn
- **Styling**: Custom CSS

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions, please open a GitHub issue or contact the development team.