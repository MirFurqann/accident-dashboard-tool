############################################################
# accident_dashboard_app.py  –  
#
# One‑page Dash dashboard + ML workbench for UK‑style
# accident CSVs (~300 k rows).  Features:
#   · CSV upload & cleaning
#   · Multi‑select filters + Find / Clear buttons
#   · Table, hourly histogram, severity pie
#   · Extra charts: Road‑type bar, accidents‑by‑light,
#     accidents‑by‑weather, sati‑temporal cube (3‑D)
#   · Single density heat‑map (Mapbox)
#   · Gradient‑Boosting ML (incl. Road Type feature)
#   · Prediction form built from dropdowns
#   · Live classification report
#   · **Advanced animated visualizations (new!)**
############################################################
import base64
import io
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.ensemble import HistGradientBoostingClassifier
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
from plotly.figure_factory import create_annotated_heatmap
from plotly.io import from_json 

# ------------------------------------------------------------------
# Dash initialisation
# ------------------------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SANDSTONE],
    suppress_callback_exceptions=True
)
app.title = "Accident GIS & ML Tool"
server = app.server

# ------------------------------------------------------------------
# Globals – will be filled after CSV upload
# ------------------------------------------------------------------
trained_model = None          # scikit‑learn model
label_encoders: dict = {}     # {col: LabelEncoder}
latest_report = ""            # str → classification_report

# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------
app.layout = dbc.Container(fluid=True, children=[
    html.H2("Traffic Accident GIS Dashboard & ML Workbench",
            className="text-center mt-3 text-primary"),
    html.Hr(),

    # ========================  UPLOAD & FILTERS  ===================
    dbc.Row([
        # ---- left: upload -----------------------------------------
        dbc.Col(width=4, children=[
            html.Label("Upload CSV file:", className="fw-bold"),
            dcc.Upload(
                id="upload-data",
                children=html.Div("Drag & Drop or Click"),
                style={
                    "width": "100%", "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "2px", "borderStyle": "dashed",
                    "borderRadius": "10px", "textAlign": "center",
                    "margin": "10px"
                },
                multiple=False
            ),
            html.Div(id="file-feedback", className="text-danger")
        ]),

        # ---- right: filters ---------------------------------------
        dbc.Col(width=8, children=[
            html.Label("Filters:", className="fw-bold"),
            # Year
            dcc.Dropdown(id="year-filter", multi=True,
                         placeholder="Year(s)",
                         style={"marginBottom": "8px"}),
            # Severity
            dcc.Dropdown(id="severity-filter", multi=True,
                         placeholder="Severity",
                         style={"marginBottom": "8px"}),
            # Light
            dcc.Dropdown(id="light-filter", multi=True,
                         placeholder="Light Condition",
                         style={"marginBottom": "8px"}),
            # Weather
            dcc.Dropdown(id="weather-filter", multi=True,
                         placeholder="Weather Condition",
                         style={"marginBottom": "8px"}),
            # Road Type
            dcc.Dropdown(id="roadtype-filter", multi=True,
                         placeholder="Road Type",
                         style={"marginBottom": "8px"}),
            dcc.Dropdown(id="hour-period-filter",
                        placeholder="Hour Period",
                        options=[
                                {"label": "All",        "value": "All"},
                                {"label": "Morning",    "value": "morning"},   # 05‑11
                                {"label": "Afternoon",  "value": "afternoon"}, # 12‑16
                                {"label": "Evening",    "value": "evening"},   # 17‑21
                                {"label": "Night",    "value": "night"},   # 21‑5
                                ],
                        style={"marginBottom": "8px"}),
            # Buttons
            dbc.Button("Find", id="btn-find", color="primary",
                       style={"marginRight": "10px"}),
            dbc.Button("Clear Filters", id="btn-clear", color="secondary")
        ])
    ]),

    # Store – dataframe & report
    dcc.Store(id="json-data"),
    dcc.Store(id="json-report"),
    # Insert here:
    html.Hr(),
    html.H4("Model Evaluation: Confusion Matrices", className="text-info"),
    dcc.Store(id="conf-matrix-hist"),
    dcc.Store(id="conf-matrix-xgb"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="fig-conf-hist"), width=6),
        dbc.Col(dcc.Graph(id="fig-conf-xgb"), width=6)
    ]),

    html.Hr(),

    # ========================  CHARTS  =====================
    dbc.Row([
        # ---- main small charts ------------------------------------
        dbc.Col(width=18, children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id="fig-hour", style={"height": "480px"})),
                dbc.Col(dcc.Graph(id="fig-sev-bar", style={"height": "480px"})),
                dbc.Col(dcc.Graph(id="fig-sev", style={"height": "480px"}))
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="fig-road",  style={"height": "380px"})),
                dbc.Col(dcc.Graph(id="fig-light", style={"height": "380px"})),
                dbc.Col(dcc.Graph(id="fig-weather", style={"height": "380px"}))
            ])
        ])
    ]),

    html.Br(),

    # ========================  MAP  =================================
    html.H5("Density Heat‑Map (Casualties)", className="text-info"),
    dcc.Graph(id="fig-map", style={"height": "520px"},
              config={"scrollZoom": True}),

    html.Hr(),

    # ====================  SPATIO‑TEMPORAL CUBE  ====================
    html.H5("Spatio‑Temporal Cube (Lat × Lon × Month)",
            className="text-info"),
    dcc.Graph(id="fig-cube", style={"height": "500px"}),

    html.Hr(),

    # ============== Advanced Animated Visualizations ================
    html.H4("Advanced Animated Visualizations", className="text-info"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="fig-animated-heatmap", style={"height": "520px"})),
        dbc.Col(dcc.Graph(id="fig-bar-race", style={"height": "520px"}))
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="fig-3d-animation", style={"height": "500px"}))
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="fig-line-animation", style={"height": "500px"}))
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="fig-period-map", style={"height": "520px"}))
    ]),

    
    html.Hr(),

    # ========================  PREDICTION  ==========================
    html.H4("Predict Severity", className="text-info"),
    dbc.Row([
        dbc.Col([
            html.Label("Hour (0‑23)"),
            dcc.Slider(id="pred-hour", min=0, max=23, step=1,
                       value=12, marks=None, tooltip={"placement": "bottom"}),
            html.Br(),
            html.Label("Vehicles Involved"),
            dcc.Input(id="pred-veh", type="number", min=1, value=1,
                      style={"width": "100%"}),
        ], width=3),

        dbc.Col([
            html.Label("Weather Condition"),
            dcc.Dropdown(id="pred-weather"),
            html.Br(),
            html.Label("Road Condition"),
            dcc.Dropdown(id="pred-road"),
            html.Br(),
            html.Label("Road Type"),
            dcc.Dropdown(id="pred-roadtype")
        ], width=3),

        dbc.Col([
            html.Label("Light Condition"),
            dcc.Dropdown(id="pred-light"),
            html.Br(),
            dbc.Button("Predict Severity (HistGradient)", id="btn-predict-severity", color="primary", className="me-2"),
            dbc.Button("Predict Severity (XGBoost)", id="btn-predict-xgb", color="success", className="me-2"),
            dbc.Button("Clear Prediction", id="btn-clear-predict", color="secondary"),
            html.Br(), html.Br(),
            html.Div(id="pred-result", className="fw-bold text-danger")
        ], width=3)
    ]),

    html.Hr(),
    
    html.H4("Predicted Pattern Overview", className="text-info"),
    dcc.Graph(id="fig-predicted-pattern", style={"height": "500px"}),

    # ===================  CLASSIFICATION REPORT  ====================
    html.Div(id="report-container", style={"display": "none"}, children=[
        html.H4("Current ML Classification Report"),
        html.Small("Note: This report is based on evaluation using test data (from uploaded dataset), not on your single prediction.",
                   className="text-muted"),
        html.Pre(id="txt-report", style={
            "whiteSpace": "pre-wrap",
            "backgroundColor": "#f8f9fa",
            "padding": "10px"
        })
    ])
])

# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------
def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename & derive standard columns."""
    df.columns = df.columns.str.strip()
    rename = {
        "Accident Date": "Date",
        "Accident_Severity": "OldSeverity",
        "Light_Conditions": "Light Condition",
        "Weather_Conditions": "Weather Condition",
        "Road_Surface_Conditions": "Road Condition",
        "Road_Type": "Road Type",
        "Number_of_Vehicles": "Vehicles Involved",
        "Number_of_Casualties": "Casualties"
    }
    for old, new in rename.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)
    return df


def _severity_from_old(s: str) -> str:
    """Slight→Low, Serious→Medium, Fatal→High."""
    if not isinstance(s, str):
        return "Low"
    s = s.lower().strip()
    if s == "slight":
        return "Low"
    if s == "serious":
        return "Medium"
    if s == "fatal":
        return "High"
    return "Low"


def _label_encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Label‑encode in‑place; store encoders."""
    label_encoders.clear()
    for c in cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        label_encoders[c] = le
    return df


# ------------------------------------------------------------------
#  CSV → DF + ML
# ------------------------------------------------------------------

def parse_csv(contents: str) -> tuple[pd.DataFrame, str]:
    global trained_model_histgrad, trained_model_xgb


    content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

    df = _clean_columns(df)

    needed = [
        "Date", "Time", "OldSeverity", "Latitude", "Longitude",
        "Weather Condition", "Road Condition", "Road Type",
        "Vehicles Involved", "Casualties"
    ]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        return None, f"Missing columns: {', '.join(miss)}"

    # ---- derive columns ------------------------------------------
    df["Severity"] = df["OldSeverity"].apply(_severity_from_old)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False, infer_datetime_format=True)
    df = df[df["Date"].notna()]
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce")
    df["Hour"] = df["Time"].dt.hour.fillna(0).astype(int)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Casualties"] = pd.to_numeric(df["Casualties"], errors="coerce").fillna(0).astype(int)
    df["Vehicles Involved"] = pd.to_numeric(df["Vehicles Involved"], errors="coerce").fillna(1).astype(int)
    df["YearMonth"] = df["Year"].astype(str).str.zfill(4) + "-" + df["Month"].astype(str).str.zfill(2)
    df["DayPeriod"] = df["Hour"].apply(lambda h: "Morning" if 5 <= h <= 11 else "Afternoon" if 12 <= h <= 16 else "Evening" if 17 <= h <= 21 else "Night")

    # ---- ML prep --------------------------------------------------
    ml_cols_cat = ["Weather Condition", "Road Condition", "Light Condition", "Road Type"]
    df_ml = _label_encode(df.copy(), ml_cols_cat)

    # Features
    X = df_ml[["Hour", "Vehicles Involved", "Weather Condition", "Road Condition", "Light Condition", "Road Type"]]

    
    # Encode Severity labels (Low, Medium, High → 0, 1, 2)
    le_sev = LabelEncoder()
    y_sev_numeric = le_sev.fit_transform(df_ml["Severity"])
    label_encoders["Severity"] = le_sev
    # ---- Train/Test Split -----------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_sev_numeric, test_size=0.25, random_state=42, stratify=y_sev_numeric
    )

    # ---- Apply SMOTE ----------------------------------------------
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    # ---- Train Severity model (HistGradientBoosting) --------------
    
    model_sev_hist = HistGradientBoostingClassifier(max_depth=7, learning_rate=0.12, max_iter=250, random_state=42)
    model_sev_hist.fit(X_train_smote, y_train_smote)
    y_pred_hist = model_sev_hist.predict(X_test)
    report_sev_hist = classification_report(y_test, y_pred_hist, target_names=le_sev.classes_)

    # ---- Train Severity model (XGBoost) -------------------------
    model_sev_xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model_sev_xgb.fit(X_train_smote, y_train_smote)
    y_pred_xgb = model_sev_xgb.predict(X_test)
    report_sev_xgb = classification_report(y_test, y_pred_xgb, target_names=le_sev.classes_)
    conf_hist = confusion_matrix(y_test, y_pred_hist)
    conf_xgb = confusion_matrix(y_test, y_pred_xgb)


    # ---- Combine reports ------------------------------------------
    combined_report = (
        "=== Severity (HistGradientBoostingClassifier) ===\n"
        + report_sev_hist +
        "\n\n=== Severity (XGBoostClassifier) ===\n"
        + report_sev_xgb
    )


    # Store both models globally
    trained_model_histgrad = model_sev_hist
    trained_model_xgb = model_sev_xgb

    # Confusion Matrices
    conf_hist = confusion_matrix(y_test, y_pred_hist)
    conf_xgb = confusion_matrix(y_test, y_pred_xgb)

    # Create annotated heatmaps for both
    conf_fig_hist = create_annotated_heatmap(
        z=conf_hist,
        x=le_sev.classes_.tolist(),
        y=le_sev.classes_.tolist(),
        colorscale="Viridis",
        showscale=True
    )
    conf_fig_hist.update_layout(title="Confusion Matrix: HistGradientBoosting")

    conf_fig_xgb = create_annotated_heatmap(
        z=conf_xgb,
        x=le_sev.classes_.tolist(),
        y=le_sev.classes_.tolist(),
        colorscale="Blues",
        showscale=True
    )
    conf_fig_xgb.update_layout(title="Confusion Matrix: XGBoost")

    # Return everything
    return df, combined_report, conf_fig_hist.to_json(), conf_fig_xgb.to_json()

def parse_csv(contents: str) -> tuple[pd.DataFrame, str, str, str]:
    global trained_model_histgrad, trained_model_xgb

    # Decode and read CSV
    content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

    df = _clean_columns(df)

    needed = [
        "Date", "Time", "OldSeverity", "Latitude", "Longitude",
        "Weather Condition", "Road Condition", "Road Type",
        "Vehicles Involved", "Casualties"
    ]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        return None, f"Missing columns: {', '.join(miss)}", "", ""

    # ------------------ Derived Columns ------------------
    df["Severity"] = df["OldSeverity"].apply(_severity_from_old)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", infer_datetime_format=True)
    df = df[df["Date"].notna()]
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce")
    df["Hour"] = df["Time"].dt.hour.fillna(0).astype(int)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Casualties"] = pd.to_numeric(df["Casualties"], errors="coerce").fillna(0).astype(int)
    df["Vehicles Involved"] = pd.to_numeric(df["Vehicles Involved"], errors="coerce").fillna(1).astype(int)
    df["YearMonth"] = df["Year"].astype(str).str.zfill(4) + "-" + df["Month"].astype(str).str.zfill(2)
    df["DayPeriod"] = df["Hour"].apply(lambda h: "Morning" if 5 <= h <= 11 else "Afternoon" if 12 <= h <= 16 else "Evening" if 17 <= h <= 21 else "Night")

    # ------------------ Feature Engineering ------------------
    df["Is_Peak_Hour"] = df["Hour"].apply(lambda h: 1 if h in list(range(7, 10)) + list(range(16, 20)) else 0)
    df["Casualties_Per_Vehicle"] = df["Casualties"] / df["Vehicles Involved"].replace(0, 1)
    df["Is_Weekend"] = df["Date"].dt.weekday.apply(lambda d: 1 if d >= 5 else 0)

    # ------------------ Encoding ------------------
    ml_cols_cat = ["Weather Condition", "Road Condition", "Light Condition", "Road Type"]
    df_ml = _label_encode(df.copy(), ml_cols_cat)

    # Feature columns
    X = df_ml[[
        "Hour", "Vehicles Involved", "Weather Condition", "Road Condition",
        "Light Condition", "Road Type", "Is_Peak_Hour", "Casualties_Per_Vehicle", "Is_Weekend"
    ]]

    # Target
    le_sev = LabelEncoder()
    y = le_sev.fit_transform(df_ml["Severity"])
    label_encoders["Severity"] = le_sev

    # ------------------ Train/Test Split ------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ------------------ Apply SMOTE ------------------
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    # ------------------ HistGradientBoosting ------------------
    model_hist = HistGradientBoostingClassifier(
        max_depth=7, learning_rate=0.1, max_iter=350, random_state=42 , class_weight='balanced'
    )
    model_hist.fit(X_train_sm, y_train_sm)
    y_pred_hist = model_hist.predict(X_test)
    report_hist = classification_report(y_test, y_pred_hist, target_names=le_sev.classes_)

    # ------------------ XGBoost ------------------
    model_xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        objective='multi:softmax',
        num_class=3,
        max_depth=10,            # less than 7 to reduce overfit
        learning_rate=0.08,      # try 0.08 to 0.12
        n_estimators=400,       # increase for better learning
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model_xgb.fit(X_train_sm, y_train_sm)
    y_pred_xgb = model_xgb.predict(X_test)
    report_xgb = classification_report(y_test, y_pred_xgb, target_names=le_sev.classes_)

    # ------------------ Confusion Matrices ------------------
    conf_hist = confusion_matrix(y_test, y_pred_hist)
    conf_xgb = confusion_matrix(y_test, y_pred_xgb)

    fig_hist = create_annotated_heatmap(
        z=conf_hist,
        x=le_sev.classes_.tolist(),
        y=le_sev.classes_.tolist(),
        colorscale="Viridis",
        showscale=True
    )
    fig_hist.update_layout(title="Confusion Matrix: HistGradientBoosting")

    fig_xgb = create_annotated_heatmap(
        z=conf_xgb,
        x=le_sev.classes_.tolist(),
        y=le_sev.classes_.tolist(),
        colorscale="Blues",
        showscale=True
    )
    fig_xgb.update_layout(title="Confusion Matrix: XGBoost")

    # ------------------ Store Models ------------------
    trained_model_histgrad = model_hist
    trained_model_xgb = model_xgb

    report_all = (
        "=== Severity (HistGradientBoostingClassifier) ===\n"
        + report_hist +
        "\n\n=== Severity (XGBoostClassifier) ===\n"
        + report_xgb
    )

    return df, report_all, fig_hist.to_json(), fig_xgb.to_json()




# ------------------------------------------------------------------
# 1. Upload CSV
# ------------------------------------------------------------------
@app.callback(
    Output("json-data", "data"),
    Output("file-feedback", "children"),
    Output("json-report", "data"),
    Output("conf-matrix-hist", "data"),  
    Output("conf-matrix-xgb",  "data"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True
)
def cb_upload(contents,filename):
    if not contents:
        return None, "No file received.", None, None, None
    # ✅ Check for valid CSV file extension
    if not filename or not filename.lower().endswith(".csv"):
        return None, "❌ Invalid file type. Please upload a CSV file.", None, None, None

    try:
        df, rep, conf_hist_json, conf_xgb_json = parse_csv(contents)
        if df is None:
            return None, rep, None, None, None

        df_json = df.to_json(date_format="iso", orient="split")

        # Optional integrity check
        from io import StringIO
        df_check = pd.read_json(StringIO(df_json), orient="split")
        print("Shape after JSON serialization:", df_check.shape)

        return df_json, "✔ File uploaded successfully.", rep, conf_hist_json, conf_xgb_json

    except Exception as e:
        return None, f"❌ Error processing file: {str(e)}", None, None, None





# ------------------------------------------------------------------
# 2. Populate filter & prediction dropdowns once data ready
# ------------------------------------------------------------------
@app.callback(
    Output("year-filter",     "options"),
    Output("severity-filter", "options"),
    Output("light-filter",    "options"),
    Output("weather-filter",  "options"),
    Output("roadtype-filter", "options"),
    Output("pred-weather",    "options"),
    Output("pred-road",       "options"),
    Output("pred-light",      "options"),
    Output("pred-roadtype",   "options"),
    Input("json-data", "data")
)
def cb_fill_opts(json_df):
    def _opts(lst):  # helper to build dropdown options
        return [{"label": "All", "value": "All"}] + \
               [{"label": str(v), "value": v} for v in lst]

    if not json_df:
        empty = _opts([])
        return empty, empty, empty, empty, empty, empty, empty, empty, empty

    df = pd.read_json(json_df, orient="split")

    years     = sorted(df["Year"].dropna().unique())
    sever     = ["Low", "Medium", "High"]
    lights    = sorted(df["Light Condition"].dropna().unique())
    weathers  = sorted(df["Weather Condition"].dropna().unique())
    roadtypes = sorted(df["Road Type"].dropna().unique())

    return (_opts(years), _opts(sever), _opts(lights),
            _opts(weathers), _opts(roadtypes),
            [{"label": v, "value": v} for v in weathers],
            [{"label": v, "value": v} for v in df["Road Condition"].dropna().unique()],
            [{"label": v, "value": v} for v in lights],
            [{"label": v, "value": v} for v in df["Road Type"].dropna().unique()])

# ------------------------------------------------------------------
# 3. Find / Clear → update charts (including original ones)
# ------------------------------------------------------------------
@app.callback(
    Output("fig-hour",   "figure"),
    Output("fig-sev-bar",    "figure"),
    Output("fig-sev",    "figure"),
    Output("fig-road",   "figure"),
    Output("fig-light",  "figure"),
    Output("fig-weather","figure"),
    Output("fig-map",    "figure"),
    Output("fig-cube",   "figure"),
    Input("btn-find",  "n_clicks"),
    Input("btn-clear", "n_clicks"),
    State("json-data",        "data"),
    State("year-filter",      "value"),
    State("severity-filter",  "value"),
    State("light-filter",     "value"),
    State("weather-filter",   "value"),
    State("roadtype-filter",  "value"),
    State("hour-period-filter","value"),
    prevent_initial_call=True
)
def cb_visuals(n_find, n_clear, json_df,
               sel_year, sel_sev, sel_light,
               sel_weather, sel_rtype, sel_hperiod):

    if not json_df:
        empty = go.Figure()
        return [], [], empty, empty, empty, empty, empty, empty, empty

    df = pd.read_json(json_df, orient="split")

    if ctx.triggered_id == "btn-clear":
        empty = go.Figure()
        return empty, empty, empty, empty, empty, empty, empty, empty

    mask = np.ones(len(df), dtype=bool)
    if sel_year      and "All" not in sel_year:      mask &= df["Year"].isin(sel_year)
    if sel_sev       and "All" not in sel_sev:       mask &= df["Severity"].isin(sel_sev)
    if sel_light     and "All" not in sel_light:     mask &= df["Light Condition"].isin(sel_light)
    if sel_weather   and "All" not in sel_weather:   mask &= df["Weather Condition"].isin(sel_weather)
    if sel_rtype     and "All" not in sel_rtype:     mask &= df["Road Type"].isin(sel_rtype)

    if sel_hperiod and sel_hperiod != "All":
        if   sel_hperiod == "morning":   mask &= df["Hour"].between(5, 11)
        elif sel_hperiod == "afternoon": mask &= df["Hour"].between(12, 16)
        elif sel_hperiod == "evening":   mask &= df["Hour"].between(17, 21)
        elif sel_hperiod == "night":  mask &= (df["Hour"] >= 21) | (df["Hour"] <= 4)

    df = df[mask]

    # ---------------  Hourly histogram  --------------------------

    # Assume df is your filtered DataFrame containing the "YearMonth" column.
    # Create aggregated copy representing overall (all months) data.
    df_all = df.copy()
    df_all["YearMonth"] = "All"

    # Concatenate the overall ("All") data with the original data.
    df_anim = pd.concat([df_all, df], ignore_index=True)

    # Define the desired order with "All" first.
    months = sorted(df["YearMonth"].unique())
    cat_order = ["All"] + months
    df_anim["YearMonth"] = pd.Categorical(df_anim["YearMonth"], categories=cat_order, ordered=True)

    # Create the animated histogram using px.histogram.
    fig_hour = px.histogram(
        df_anim,
        x="Hour",
        nbins=24,
        animation_frame="YearMonth",  # Animation across "All" and each month.
        title="Animated Accidents by Hour",
        labels={"Hour": "Hour of Day", "count": "Accident Count"},
        template="plotly_white"
    )
    fig_hour.update_traces(marker_color='royalblue')
    fig_hour.update_layout(
        xaxis=dict(dtick=1),
        title_x=0.5,
        margin=dict(t=40, b=40, l=20, r=20)
    )

# ---------------  Severity Bar Chart   --------------------------

    # Assume df is your filtered DataFrame and it includes a "DayPeriod" column
    # with values like "Morning", "Afternoon", "Evening", "Night".

    # Aggregate severity counts per DayPeriod.
    sev_dp = df.groupby(["DayPeriod", "Severity"]).size().reset_index(name="Count")

    # Compute overall ("All") severity counts.
    sev_all = df.groupby("Severity").size().reset_index(name="Count")
    sev_all["DayPeriod"] = "All"

    # Combine the two DataFrames.
    sev_anim = pd.concat([sev_all, sev_dp], ignore_index=True)

    # Define the order for DayPeriod including "All".
    day_order = ["All", "Morning", "Afternoon", "Evening", "Night"]
    sev_anim["DayPeriod"] = pd.Categorical(sev_anim["DayPeriod"], categories=day_order, ordered=True)

    # Create the animated bar chart.
    fig_sev_bar = px.bar(
        sev_anim,
        x="Severity",
        y="Count",
        color="Severity",
        animation_frame="DayPeriod",  # Animate across "All", "Morning", etc.
        title="Severity Distribution by Day Period",
        template="plotly_white",
        text_auto=True
    )
    fig_sev_bar.update_layout(
        title_x=0.5,
        xaxis_title="Severity",
        yaxis_title="Count",
        margin=dict(t=40, b=40, l=20, r=20)
    )

    # ---------------  Severity pie chart  ------------------------

    # Aggregate severity counts per YearMonth.
    sev_df = df.groupby(["YearMonth", "Severity"]).size().reset_index(name="Count")
    # Aggregate overall severity counts ("All").
    sev_all = df.groupby("Severity").size().reset_index(name="Count")
    sev_all["YearMonth"] = "All"

    # Concatenate both aggregated datasets.
    sev_anim = pd.concat([sev_all, sev_df], ignore_index=True)

    # Ensure the "YearMonth" column follows the desired order.
    sev_anim["YearMonth"] = pd.Categorical(sev_anim["YearMonth"], categories=cat_order, ordered=True)

    # Create frames for each slider value.
    frames = [
        go.Frame(
            data=[go.Pie(
                labels=sev_anim[sev_anim["YearMonth"] == period]["Severity"],
                values=sev_anim[sev_anim["YearMonth"] == period]["Count"],
                textinfo='label+percent'
            )],
            name=period,
            layout=go.Layout(title_text=f"Severity Distribution: {period}")
        )
        for period in cat_order
    ]

    # Initial data using the "All" aggregated counts.
    initial = sev_anim[sev_anim["YearMonth"] == "All"]

    fig_sev = go.Figure(
        data=[go.Pie(
            labels=initial["Severity"],
            values=initial["Count"],
            textinfo='label+percent'
        )],
        layout=go.Layout(
            title_text="Severity Distribution: All",
            title_x=0.5,
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 1500, "redraw": True},
                                        "fromcurrent": True, "mode": "immediate"}]
                    },
                    {
                        "label": "Stop",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}]
                    }
                ]
            }]
        ),
        frames=frames
    )

    # Add a slider with an "All" option plus individual months.
    fig_sev.update_layout(
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Time Period: "},
            "pad": {"t": 50},
            "steps": [
                {
                    "method": "animate",
                    "args": [[period], {"mode": "immediate", "frame": {"duration": 1500, "redraw": True}}],
                    "label": period
                }
                for period in cat_order
            ]
        }]
    )


    # ---------------  Road‑type bar chart  ------------------------
   # Aggregate data to count accidents per Road Type
    df_road = df.groupby("Road Type").size().reset_index(name="Count")

    # Create a bar chart using the aggregated data
    fig_road = px.bar(
        df_road,
        x="Road Type",
        y="Count",  # now each bar height is the count of accidents
        title="Road‑Type Distribution",
        labels={"Road Type": "Road Type", "Count": "Accident Count"},
        template="plotly_white"
    )
    fig_road.update_layout(xaxis_tickangle=45, title_x=0.5, plot_bgcolor="white")


    # ---------------  Accidents by Light  -------------------------
    fig_light = px.histogram(df, x="Light Condition",
                             title="Accidents by Light Condition",
                             labels={"Light Condition": "Light Condition", "count": "Count"})
    fig_light.update_layout(xaxis_tickangle=45, plot_bgcolor="white", title_x=0.5)

    # ---------------  Accidents by Weather  -----------------------
    fig_weather = px.histogram(df, x="Weather Condition",
                               title="Accidents by Weather",
                               labels={"Weather Condition": "Weather Condition", "count": "Count"})
    fig_weather.update_layout(xaxis_tickangle=45, plot_bgcolor="white", title_x=0.5)

    # ---------------  Density map  ---------------------------------
    if df.empty:
        fig_map = go.Figure()
        fig_map.add_annotation(text="No data", showarrow=False)
    else:
        fig_map = px.density_mapbox(
            df, lat="Latitude", lon="Longitude", z="Casualties",
            radius=15, center=dict(lat=51.5, lon=-0.1),
            zoom=6, mapbox_style="open-street-map"
        )
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # ---------------  Spatio‑temporal 3D cube  --------------------
    if df.empty:
        fig_cube = go.Figure()
    else:
        fig_cube = px.scatter_3d(
            df, x="Longitude", y="Latitude", z="Month",
            color="Severity", size="Casualties",
            title="Lat / Lon / Month"
        )
        fig_cube.update_layout(margin={"r":0,"t":40,"l":0,"b":0})

    return  fig_hour, fig_sev_bar, fig_sev, fig_road, fig_light, fig_weather, fig_map, fig_cube


# ------------------------------------------------------------------
# 4. Advanced Animated Visualizations Callback – FILTER‑AWARE
# ------------------------------------------------------------------
@app.callback(
    Output("fig-animated-heatmap", "figure"),
    Output("fig-bar-race", "figure"),
    Output("fig-3d-animation", "figure"),
    Output("fig-line-animation", "figure"),
    Output("fig-period-map",      "figure"),
    Input("btn-find",       "n_clicks"),
    Input("btn-clear",      "n_clicks"),
    State("json-data",      "data"),
    State("year-filter",    "value"),
    State("severity-filter","value"),
    State("light-filter",   "value"),
    State("weather-filter", "value"),
    State("roadtype-filter","value"),
    prevent_initial_call=True
)
def cb_advanced_visuals( n_find, n_clear,json_df,
                        sel_year, sel_sev, sel_light,
                        sel_weather, sel_rtype):
    empty = go.Figure()
    if not json_df:
        return empty, empty, empty, empty, empty

    df = pd.read_json(json_df, orient="split")

    # Reset filters if “Clear” clicked
    if ctx.triggered_id == "btn-clear":
        empty = go.Figure()
        return empty, empty, empty, empty, empty

    # Apply filters
    def _filt(col, sel):
        return df if not sel or "All" in sel else df[df[col].isin(sel)]
    df = _filt("Year", sel_year)
    df = _filt("Severity", sel_sev)
    df = _filt("Light Condition", sel_light)
    df = _filt("Weather Condition", sel_weather)
    df = _filt("Road Type", sel_rtype)

    if df.empty:
        return empty, empty, empty, empty,empty

    # ---------- Animated Time‑Series Heat‑Map ----------
    fig_anim_heatmap = px.density_mapbox(
        df, lat="Latitude", lon="Longitude", z="Casualties",
        animation_frame="Hour", radius=15,
        center={"lat": 51.5, "lon": -0.1}, zoom=6,
        mapbox_style="open-street-map",
        title="Animated Heat‑Map by Hour"
    ).update_layout(margin=dict(r=0, t=0, l=0, b=0))

    # ---------- Animated Bar Chart Race ----------
    df_bar = df.groupby(["Year", "Road Type"])["Casualties"].sum().reset_index()
    fig_bar_race = px.bar(
        df_bar, x="Road Type", y="Casualties", color="Road Type",
        animation_frame="Year", animation_group="Road Type",
        title="Top Road Types by Casualties Over Years"
    )

    # ---------- Animated 3‑D Scatter ----------
    fig_3d_anim = px.scatter_3d(
        df, x="Longitude", y="Latitude", z="Hour",
        color="Severity", size="Casualties",
        animation_frame="YearMonth",
        hover_data=["District"] if "District" in df.columns else None,
        title="3‑D Accident Pattern by Time"
    ).update_layout(margin=dict(r=0, t=40, l=0, b=0))

    # ---------- Animated Line Chart ----------
    df_line = df.groupby(["Year", "Hour", "Severity"]).size().reset_index(name="Count")
    fig_line_anim = px.line(
        df_line, x="Hour", y="Count", color="Severity",
        animation_frame="Year",
        title="Accidents by Hour and Severity Over Years"
    )
 # ----------  Animated GIS Map by Day Period (Morning/Afternoon/Evening)
    # Instead of a density heatmap, we now build a scatter_mapbox which
    # plots individual accident points. (Night is left out, per your request.)
    df_period = df[df["DayPeriod"].isin(["Morning", "Afternoon", "Evening", "Night"])]
    # For proper ordering in the animation, set a category order.
    period_order = {"Morning": 1, "Afternoon": 2, "Evening": 3, "Night": 4}
    df_period = df_period.copy()
    df_period["PeriodOrder"] = df_period["DayPeriod"].map(period_order)
    # Sort by the custom order (helps the animation ordering)
    df_period.sort_values("PeriodOrder", inplace=True)

    fig_period_map = px.scatter_mapbox(
        df_period,
        lat="Latitude", lon="Longitude",
        # You may use Casualties to adjust marker size or use Severity for color.
        size="Casualties",
        color="Severity",
        animation_frame="DayPeriod",  # frames: Morning, Afternoon, Evening
        title="Animated GIS Map by Day Period",
        zoom=6,
        center={"lat": 51.5, "lon": -0.1},
        mapbox_style="open-street-map",
        hover_data=["Road Type", "Weather Condition", "Light Condition"]
    )
    fig_period_map.update_layout(
        margin=dict(r=0, t=40, l=0, b=0),
        transition={"duration": 500}
    )

    return fig_anim_heatmap, fig_bar_race, fig_3d_anim, fig_line_anim, fig_period_map



# ------------------------------------------------------------------
# 5. Prediction Callback
# ------------------------------------------------------------------
@app.callback(
    Output("pred-result", "children"),
    Output("fig-predicted-pattern", "figure"),
    Input("btn-predict-severity", "n_clicks"),
    Input("btn-predict-xgb", "n_clicks"),
    Input("btn-clear-predict", "n_clicks"),
    State("json-data", "data"),
    State("pred-hour", "value"),
    State("pred-veh", "value"),
    State("pred-weather", "value"),
    State("pred-road", "value"),
    State("pred-light", "value"),
    State("pred-roadtype", "value"),
    prevent_initial_call=True
)
def cb_predict_buttons(n_sev, n_rtype, n_clear, json_df, hour, veh, weather, road, light,roadtype):
    global trained_model_histgrad, trained_model_xgb


    triggered = ctx.triggered_id

    if triggered == "btn-clear-predict":
        empty_fig = go.Figure().add_annotation(text="Prediction cleared", showarrow=False)
        return "", empty_fig

    if not json_df:
        no_data_fig = go.Figure().add_annotation(text="No matching data", showarrow=False)
        return "No models – upload data first.", no_data_fig

    df = pd.read_json(json_df, orient="split")

    def resolve_value(val, col):
        if val == "All" or val is None:
            return df[col].mode()[0]
        return val

    weather = resolve_value(weather, "Weather Condition")
    road = resolve_value(road, "Road Condition")
    light = resolve_value(light, "Light Condition")
    roadtype = resolve_value(roadtype, "Road Type")

    row = {
        "Hour": hour,
        "Vehicles Involved": veh,
        "Weather Condition": weather,
        "Road Condition": road,
        "Light Condition": light,
        "Road Type": roadtype
    }

    for col in ["Weather Condition", "Road Condition", "Light Condition", "Road Type"]:
        le = label_encoders.get(col)
        if le:
            try:
                row[col] = le.transform([row[col]])[0]
            except ValueError:
                row[col] = 0
        else:
            row[col] = 0
    # --- Derived features based on inputs ---

    # Is_Peak_Hour: 7–9 AM and 4–7 PM
    is_peak_hour = 1 if hour in list(range(7, 10)) + list(range(16, 20)) else 0

    # Casualties_Per_Vehicle: assume 1 casualty by default (safe constant)
    casualties = 1
    casualties_per_vehicle = casualties / veh if veh else 1

    # Is_Weekend: use current weekday if no date is selected (defaults to today)
    is_weekend = 1 if datetime.now().weekday() >= 5 else 0
    X_input = np.array([[
                            hour,
                            veh,
                            row["Weather Condition"],
                            row["Road Condition"],
                            row["Light Condition"],
                            row["Road Type"],
                            is_peak_hour,
                            casualties_per_vehicle,
                            is_weekend
    ]])

    if triggered == "btn-predict-severity":
        pred_code = trained_model_histgrad.predict(X_input)[0]
        pred_label = label_encoders["Severity"].inverse_transform([pred_code])[0]
        result = f"🔹 Predicted Severity (HistGradient) → **{pred_label}**"


    elif triggered == "btn-predict-xgb":
        pred_code = trained_model_xgb.predict(X_input)[0]
        pred_label = label_encoders["Severity"].inverse_transform([pred_code])[0]
        result = f"🔹 Predicted Severity (XGBoost) → **{pred_label}**"



    else:
        result = "Unknown action."

    # Matching pattern figure
    df_match = df.copy()
    if df_match.empty:
        fig_match = go.Figure().add_annotation(text="No matching data found", showarrow=False)
    else:
        agg = df_match.groupby(["Hour", "Road Type", "Severity", "Road Condition", "Light Condition", "Weather Condition"]) \
                      .size().reset_index(name="Count")

        fig_match = px.scatter(
            agg,
            x="Road Type",
            y="Severity",
            size="Count",
            color="Road Condition",
            animation_frame="Hour",
            size_max=60,
            hover_data={"Road Condition": True, "Light Condition": True, "Weather Condition": True, "Count": True},
            title="Pattern Overview: Severity vs. Road Type (bubble size = count, slider = hour)",
            template="plotly_white"
        )
        fig_match.update_layout(
            title_x=0.5,
            margin=dict(t=40, b=40, l=20, r=20),
            xaxis_title="Road Type",
            yaxis_title="Severity",
            legend_title="Road Condition"
        )

    return result, fig_match






# ------------------------------------------------------------------
# 6. Display Classification Report
# ------------------------------------------------------------------
@app.callback(
    Output("report-container", "style"),
    Output("txt-report", "children"),
    Input("json-report", "data")
)
def cb_show_report(rep):
    if not rep:
        return {"display": "none"}, ""
    # Optionally remove timestamp from display:
    lines = rep.split("\n")
    clean_rep = "\n".join(lines[1:]) if "Updated:" in lines[0] else rep
    return {"display": "block"}, clean_rep


@app.callback(
    Output("fig-conf-hist", "figure"),
    Output("fig-conf-xgb", "figure"),
    Input("conf-matrix-hist", "data"),
    Input("conf-matrix-xgb", "data"),
    prevent_initial_call=True
)
def cb_show_conf(data_hist, data_xgb):
    if not data_hist or not data_xgb:
        no_data_fig = go.Figure().add_annotation(text="No data", showarrow=False)
        return no_data_fig, no_data_fig

    fig_hist = from_json(data_hist)
    fig_xgb  = from_json(data_xgb)

    return fig_hist, fig_xgb


# ------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)