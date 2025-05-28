# Accident Dashboard Tool

This is the solution part of Furqan Ul Islam's Bachelor Thesis.  
The prototype is developed using Python, and the dataset used for this tool is a **modified version** (Modified by Furqan Ul Islam) of the original dataset from [Kaggle](https://www.kaggle.com/datasets/nextmillionaire/car-accident-dataset).


## Abstract of the Thesis

This bachelor thesis presents the development of an advanced, interactive web-based visualization tool designed to analyze the correlation between daytime conditions and traffic accidents.  
Traditional traffic accident analysis relies on static visualization methods, which fail to capture temporal accident trends and provide real-time insights.

Addressing these limitations, this research introduces a dynamic, data-driven platform that integrates critical factors—including road traffic, weather conditions, road density, driver behaviour, and vehicle characteristics—to offer a more comprehensive and interactive understanding of accident patterns.  

To overcome deficiencies in existing methodologies, this study employs cutting-edge visualization techniques such as animated heatmaps, time-series analysis, and geospatial mapping powered by Plotly Dash.  
Furthermore, statistical modelling—incorporating regression analysis and correlation studies—is utilized to quantify the impact of environmental factors on accident risks.

Machine learning models (HistGradientBoostingClassifier and XGBoost Classifier) are implemented using Scikit-learn and XGBoost, analyzing historical accident data to predict risk factors based on time-of-day variations and weather conditions.  
Both classifiers are evaluated using standard classification metrics such as precision, recall, and F1-score, along with confusion matrix visualizations that offer clear insights into prediction accuracy across severity classes (Low, Medium, High).

---

## Overview

This project is a one-page interactive Dash-based dashboard and machine learning workbench designed for analyzing UK-style road traffic accident datasets (~300,000 rows).  
It supports both geographical and statistical analysis and allows for real-time predictions based on user input.  
It includes features like:

- CSV upload & real-time preprocessing
- Dynamic filters and visual exploration
- Animated heatmaps, bar races, and 3D GIS visualizations
- Live ML prediction for severity using Histgradient and XGBoost (Machine Learning Algorithms) 
- Classification report and confusion matrix display

---

## Data Preprocessing & Class Balancing

Before being used in the dashboard, the original dataset is first preprocessed and class-balanced using a custom Python script.  
This script adjusts the distribution of severity classes (Low, Medium, High) and adds a small bias boost to underrepresented road types (such as types 0 and 3).  
The script ensures a realistic yet balanced representation of accident severity across different road types, improving machine learning prediction performance.  

This preprocessing step creates the `balanced_realistic_accident_dataset.csv` file used in the tool.  
The original unbalanced dataset is also included as `Original_Road_Accident_Data.csv`.

---

## Getting Started

### Clone the repository
  git clone https://github.com/MirFurqann/accident-dashboard-tool.git
  cd accident-dashboard-tool
  
### Set up environment using conda
  conda env create -f environment.yml
  conda activate accident-dashboard-env
  Or simply double-click setup.bat (Windows users).

### Running the Tool
  python accident_dashboard_app.py
  Then open your browser at:
  http://127.0.0.1:8050

### Git LFS Note (CSV files)
      The .csv files in this repository are stored using Git LFS due to their large size.
      
      🔹 If you cloned the repo and only see small .txt pointer files like this:
      
      version https://git-lfs.github.com/spec/v1
      oid sha256:...
      size 66123456
      It means GitHub gave you a Git LFS pointer, not the actual dataset.
      
      To download the real CSV files:
  ### Option 1 (Best): Use Git LFS properly
      
      git lfs install
      git lfs pull
      This will download the full .csv data files into your project folder.
      
  ### Option 2 (Quick workaround): Manual Notepad method
      
      If you downloaded the file via GitHub’s browser and it opens as plain text:
      
      Right-click the downloaded file → Open with Notepad
      
      Delete the LFS pointer content if needed
      
      If you have a copy of the real CSV content (e.g., from Kaggle or local), paste it in
      
      Click File > Save As
      
      Choose "All Files" in the dropdown
      
      Rename the file to:
      balanced_realistic_accident_dataset.csv
      or
      Original_Road_Accident_Data.csv
      
      Make sure the file type is saved as .csv, not .txt
      
      This is only a workaround. Using git lfs pull for full data integrity and automation is recommended.

