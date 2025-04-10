Accident Dashboard Tool
=======================
This is a Solution part of the Furqan Ul Islam's  Bachelor Thesis. 
The prototype is developed using the Python Lanuage and the dataset used for this prototype is modified version (Modified By Furqan Ul Islam, Author of this tool) of the original dataset taken from https://www.kaggle.com/datasets/nextmillionaire/car-accident-dataset.


Abstarct of The Thesis : This bachelor thesis presents the development of an advanced, interactive web-based visualization tool designed to analyze the correlation between daytime conditions and traffic accidents.
Traditional traffic accident analysis relies on static visualization methods, which fail to capture temporal accident trends and provide real-time insights.
Addressing these limitations, this research introduces a dynamic, data-driven platform that integrates critical factors—including road traffic, weather conditions, road density, driver behaviour, and vehicle characteristics—to offer a more comprehensive and interactive understanding of accident patterns. 
To overcome deficiencies in existing methodologies, this study employs cutting-edge visualization techniques such as animated heatmaps, time-series analysis, and geospatial mapping powered by Plotly Dash. 
Furthermore, statistical modelling—incorporating regression analysis and correlation studies—is utilized to quantify the impact of environmental factors on accident risks. 
The proposed tool is developed as a one-page Python Dash application, utilizing HTML, CSS, and Bootstrap for layout styling, and Plotly for interactive frontend visualizations, while Pandas and NumPy handle backend data preprocessing. 
Machine learning models(HistGradientBoostingClassifier)  are implemented using Scikit-learn, analysing historical accident data to predict risk factors based on time-of-day variations and weather conditions. 
Rigorous testing ensures high performance, precision, recall, and usability, with real-time evaluation metrics and user feedback guiding refinements. 

Overview
--------
This project is a one-page interactive Dash-based dashboard and machine learning workbench designed for analyzing UK-style road traffic accident datasets (~60,000 rows).
It supports both geographical and statistical analysis and allows for real-time predictions based on user input.
It includes features like data upload, preprocessing, dynamic filtering, a variety of visual analytics, animated visualizations, and ML-based severity and road type predictions.

Key Features
------------
📤 CSV Upload & Auto-Cleaning

🔍 Multi-Select Filters: Year, Severity, Weather, Light, Road Type, Hour Period

📊 Animated Hourly Histograms & Severity Charts

📌 Interactive Visuals:

Road Type, Light, Weather Distribution

Spatio-Temporal 3D Cube

GIS Density Heatmap

🧠 ML Predictions (Severity + Road Type)

📈 Live Classification Report with Evaluation Metrics

🌀 Advanced Animated Visualizations:

Animated Bar Race, 3D Scatter, Line Charts by Year & Hour

GIS Map by Day Period (Morning, Afternoon, Evening, Night)

Setup Instructions
------------------

1. Clone the Repository

```bash
git clone https://github.com/MirFurqann/accident-dashboard-tool.git
cd accident-dashboard-tool
2. Install Dependencies

If you're using pip, install the required packages with:

            pip install -r requirements.txt

Or if you're using Conda (recommended for Visual Studio / Anaconda users):

            conda env create -f environment.yml
            conda activate accident-dashboard-env

3. Optional (Windows users only):

You can run the setup.bat file by double-clicking it. This will:

Create a virtual environment

Activate it

Install required packages

4. Run the Application

Once everything is installed, start the app by running:

python accident_dashboard_app.py

Then open your browser and go to:
http://127.0.0.1:8050
