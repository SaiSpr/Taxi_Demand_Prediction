import zipfile 
from datetime import datetime, timedelta

import requests
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import pydeck as pdk

from src.inference import (
    load_predictions_from_store,
    load_batch_of_features_from_store
)
#######################################################
from datetime import datetime, timedelta

import hopsworks
from hsfs.feature_store import FeatureStore
import pandas as pd
import numpy as np

import src.config as config
from src.feature_store_api import get_feature_store

HOPSWORKS_API_KEY = 'IB2b1Snz4GTx8CtU.XbeQ6PKTngkgDlSAmcgIMbZZKQErPI473MTYm9g2HC5maLTX9PyePJMWHcLWGbCh'

HOPSWORKS_PROJECT_NAME = 'Time_Series_NYC'

# try:
#     HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
# except:
#     raise Exception('Create an .env file on the project root with the HOPSWORKS_API_KEY')

FEATURE_GROUP_NAME = 'new_time_series_feature_group'
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = 'new_time_series_feature_view'
FEATURE_VIEW_VERSION = 1
MODEL_NAME = "taxi_demand_predictor_next_hour"
MODEL_VERSION = 1

# 

# added for monitoring purposes
FEATURE_GROUP_MODEL_PREDICTIONS = 'model_predictions_feature_group'
FEATURE_VIEW_MODEL_PREDICTIONS = 'model_predictions_feature_view'
FEATURE_VIEW_MONITORING = 'predictions_vs_actuals_for_monitoring_feature_view'

# number of historical values our model needs to generate predictions
N_FEATURES = 24 * 28

# maximum Mean Absolute Error we allow our production model to have
MAX_MAE = 4.0

def get_hopsworks_project() -> hopsworks.project.Project:

    return hopsworks.login(
        project='Time_Series_NYC',
        api_key_value='IB2b1Snz4GTx8CtU.XbeQ6PKTngkgDlSAmcgIMbZZKQErPI473MTYm9g2HC5maLTX9PyePJMWHcLWGbCh'
    )

# def get_feature_store() -> FeatureStore:
    
#     project = get_hopsworks_project()
#     return project.get_feature_store()


def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """"""
    # past_rides_columns = [c for c in features.columns if c.startswith('rides_')]
    predictions = model.predict(features)

    results = pd.DataFrame()
    results['pickup_location_id'] = features['pickup_location_id'].values
    results['predicted_demand'] = predictions.round(0)
    
    return results


def load_batch_of_features_from_store(
    current_date: datetime,    
) -> pd.DataFrame:
    """Fetches the batch of features used by the ML system at `current_date`

    Args:
        current_date (datetime): datetime of the prediction for which we want
        to get the batch of features

    Returns:
        pd.DataFrame: 3 columns:
            - `pickup_hour`
            - `rides`
            - `pickup_location_id`
    """
    n_features = config.N_FEATURES

    feature_store = get_feature_store()

    # read time-series data from the feature store
    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=28)
    print(f'Fetching data from {fetch_data_from} to {fetch_data_to}')
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION
    )
    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to + timedelta(days=1))
    )
    ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]

    # validate we are not missing data in the feature store
    location_ids = ts_data['pickup_location_id'].unique()
    #assert len(ts_data) == n_features*len(location_ids), \
    #"Time-series data is not complete. Make sure your feature pipeline is up and runnning."
    
    # sort data by location and time
    ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)
    # print(f'{ts_data=}')

    # transpose time-series data as a feature vector, for each `pickup_location_id`
    x = np.ndarray(shape=(len(location_ids), n_features), dtype=np.float32)
    for i, location_id in enumerate(location_ids):
        ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, :]
        ts_data_i = ts_data_i.sort_values(by=['pickup_hour'])       
        x[i, :len(ts_data_i['rides'].values)] = ts_data_i['rides'].values


    # numpy arrays to Pandas dataframes
    features = pd.DataFrame(
        x,
        columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(n_features))]
    )
    features['pickup_hour'] = current_date
    features['pickup_location_id'] = location_ids
    features.sort_values(by=['pickup_location_id'], inplace=True)

    return features
    

def load_model_from_registry():
    
    import joblib
    from pathlib import Path

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    model = model_registry.get_model(
        name=config.MODEL_NAME,
        version=config.MODEL_VERSION,
    )  
    
    model_dir = model.download()
    model = joblib.load(Path(model_dir)  / 'model.pkl')
       
    return model

def load_predictions_from_store(
        from_pickup_hour: datetime,
        to_pickup_hour: datetime) -> pd.DataFrame:
    """
    Connects to the feature store and retrieves model predictions for all
    `pickup_location_id`s and for the time period from `from_pickup_hour`
    to `to_pickup_hour`

    Args:
        from_pickup_hour (datetime): min datetime (rounded hour) for which we want to get
        predictions

        to_pickup_hour (datetime): max datetime (rounded hour) for which we want to get
        predictions

    Returns:
        pd.DataFrame: 3 columns:
            - `pickup_location_id`
            - `predicted_demand`
            - `pickup_hour`
    """
    from src.feature_store_api import get_feature_store
    import src.config as config

    feature_store = get_feature_store()

    predictiong_fg = feature_store.get_feature_group(
        name=config.FEATURE_GROUP_MODEL_PREDICTIONS,
        version=1,
    )

    try:
        # create feature view as it does not exist yet
        feature_store.create_feature_view(
            name=config.FEATURE_VIEW_MODEL_PREDICTIONS,
            version=1,
            query=predictiong_fg.select_all()
        )
    except:
        print(f'Feature view {config.FEATURE_VIEW_MODEL_PREDICTIONS} \
              already existed. Skipped creation.')
        
    predictions_fv = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_MODEL_PREDICTIONS,
        version=1
    )
    
    print(f'Fetching predictions for `pickup_hours` between {from_pickup_hour}  and {to_pickup_hour}')
    predictions = predictions_fv.get_batch_data(
        start_time=from_pickup_hour - timedelta(days=1),
        end_time=to_pickup_hour + timedelta(days=1)
    )
    predictions = predictions[predictions.pickup_hour.between(
        from_pickup_hour, to_pickup_hour)]

    # sort by `pick_up_hour` and `pickup_location_id`
    predictions.sort_values(by=['pickup_hour', 'pickup_location_id'], inplace=True)

    return predictions
#########################################################
# from src.paths import DATA_DIR
# from src.plot import plot_one_sample

#############################################################################################
from typing import Optional, List
from datetime import timedelta

import pandas as pd
import plotly.express as px 

def plot_one_sample(
    example_id: int,
    features: pd.DataFrame,
    targets: Optional[pd.Series] = None,
    predictions: Optional[pd.Series] = None
):
    """"""
    features_ = features.iloc[example_id]
    
    if targets is not None:
        target_ = targets.iloc[example_id]
    else:
        target_ = None
    
    ts_columns = [c for c in features.columns if c.startswith('rides_previous_')]
    ts_values = [features_[c] for c in ts_columns] + [target_]
    ts_dates = pd.date_range(
        features_['pickup_hour'] - timedelta(hours=len(ts_columns)),
        features_['pickup_hour'],
        freq='H'
    )
    
    # line plot with past values
    title = f'Pick up hour={features_["pickup_hour"]}, location_id={features_["pickup_location_id"]}'
    fig = px.line(
        x=ts_dates, y=ts_values,
        template='plotly_dark',
        markers=True, title=title
    )
    
    if targets is not None:
        # green dot for the value we wanna predict
        fig.add_scatter(x=ts_dates[-1:], y=[target_],
                        line_color='green',
                        mode='markers', marker_size=10, name='actual value') 
        
    if predictions is not None:
        # big red X for the predicted value, if passed
        prediction_ = predictions.iloc[example_id]
        fig.add_scatter(x=ts_dates[-1:], y=[prediction_],
                        line_color='red',
                        mode='markers', marker_symbol='x', marker_size=15,
                        name='prediction')             
    return fig


def plot_ts(
    ts_data: pd.DataFrame,
    locations: Optional[List[int]] = None
    ):
    """
    Plot time-series data
    """
    ts_data_to_plot = ts_data[ts_data.pickup_location_id.isin(locations)] if locations else ts_data

    fig = px.line(
        ts_data,
        x="pickup_hour",
        y="rides",
        color='pickup_location_id',
        template='none',
    )

    fig.show()
####################################################################################################

st.set_page_config(layout="wide")



# title
current_date = datetime.strptime('2023-09-20 10:59:59', '%Y-%m-%d %H:%M:%S')
# current_date = pd.to_datetime(datetime.utcnow()).floor('H')
st.title(f'Taxi demand prediction 🚕')
st.header(f'{current_date} UTC')

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 6


def load_shape_data_file() -> gpd.geodataframe.GeoDataFrame:
    """
    Fetches remote file with shape data, that we later use to plot the
    different pickup_location_ids on the map of NYC.

    Raises:
        Exception: when we cannot connect to the external server where
        the file is.

    Returns:
        GeoDataFrame: columns -> (OBJECTID	Shape_Leng	Shape_Area	zone	LocationID	borough	geometry)
    """
    # download zip file
    URL = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip'
    response = requests.get(URL)
    path = DATA_DIR / f'taxi_zones.zip'
    if response.status_code == 200:
        open(path, "wb").write(response.content)
    else:
        raise Exception(f'{URL} is not available')

    # unzip file
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR / 'taxi_zones')

    # load and return shape file
    return gpd.read_file(DATA_DIR / 'taxi_zones/taxi_zones.shp').to_crs('epsg:4326')


# @st.cache_data
def _load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    """Wrapped version of src.inference.load_batch_of_features_from_store, so
    we can add Streamlit caching

    Args:
        current_date (datetime): _description_

    Returns:
        pd.DataFrame: n_features + 2 columns:
            - `rides_previous_N_hour`
            - `rides_previous_{N-1}_hour`
            - ...
            - `rides_previous_1_hour`
            - `pickup_hour`
            - `pickup_location_id`
    """
    return load_batch_of_features_from_store(current_date)

# @st.cache_data
def _load_predictions_from_store(
    from_pickup_hour: datetime,
    to_pickup_hour: datetime
    ) -> pd.DataFrame:
    """
    Wrapped version of src.inference.load_predictions_from_store, so we
    can add Streamlit caching

    Args:
        from_pickup_hour (datetime): min datetime (rounded hour) for which we want to get
        predictions

        to_pickup_hour (datetime): max datetime (rounded hour) for which we want to get
        predictions

    Returns:
        pd.DataFrame: 2 columns: pickup_location_id, predicted_demand
    """
    return load_predictions_from_store(from_pickup_hour, to_pickup_hour)

with st.spinner(text="Downloading shape file to plot taxi zones"):
    geo_df = load_shape_data_file()
    st.sidebar.write('✅ Shape file was downloaded ')
    progress_bar.progress(1/N_STEPS)

with st.spinner(text="Fetching model predictions from the store"):
    predictions_df = _load_predictions_from_store(
        from_pickup_hour=current_date - timedelta(hours=1),
        to_pickup_hour=current_date
        )
    st.sidebar.write('✅ Model predictions arrived')
    progress_bar.progress(2/N_STEPS)

# Here we are checking the predictions for the current hour have already been computed
# and are available
next_hour_predictions_ready = \
    True if predictions_df[predictions_df.pickup_hour == current_date].empty else True
prev_hour_predictions_ready = \
    True if predictions_df[predictions_df.pickup_hour == (current_date - timedelta(hours=1))].empty else True

if next_hour_predictions_ready:
    # predictions for the current hour are available
    predictions_df = predictions_df[predictions_df.pickup_hour == current_date]
elif prev_hour_predictions_ready:
    # predictions for current hour are not available, so we use previous hour predictions
    predictions_df = predictions_df[predictions_df.pickup_hour == (current_date - timedelta(hours=1))]
    current_date = current_date - timedelta(hours=1)
    st.subheader('⚠️ The most recent data is not yet available. Using last hour predictions')
else:
    raise Exception('Features are not available for the last 2 hours. Is your feature \
                    pipeline up and running? 🤔')


with st.spinner(text="Preparing data to plot"):

    def pseudocolor(val, minval, maxval, startcolor, stopcolor):
        """
        Convert value in the range minval...maxval to a color in the range
        startcolor to stopcolor. The colors passed and the the one returned are
        composed of a sequence of N component values.

        Credits to https://stackoverflow.com/a/10907855
        """
        f = float(val-minval) / (maxval-minval)
        return tuple(f*(b-a)+a for (a, b) in zip(startcolor, stopcolor))
        
    df = pd.merge(geo_df, predictions_df,
                  right_on='pickup_location_id',
                  left_on='LocationID',
                  how='inner')
    
    BLACK, GREEN = (0, 0, 0), (0, 255, 0)
    df['color_scaling'] = df['predicted_demand']
    max_pred, min_pred = df['color_scaling'].max(), df['color_scaling'].min()
    df['fill_color'] = df['color_scaling'].apply(lambda x: pseudocolor(x, min_pred, max_pred, BLACK, GREEN))
    progress_bar.progress(3/N_STEPS)

with st.spinner(text="Generating NYC Map"):

    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=40.7831,
        longitude=-73.9712,
        zoom=11,
        max_zoom=16,
        pitch=45,
        bearing=0
    )

    geojson = pdk.Layer(
        "GeoJsonLayer",
        df,
        opacity=0.25,
        stroked=False,
        filled=True,
        extruded=False,
        wireframe=True,
        get_elevation=10,
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255],
        auto_highlight=True,
        pickable=True,
    )

    tooltip = {"html": "<b>Zone:</b> [{LocationID}]{zone} <br /> <b>Predicted rides:</b> {predicted_demand}"}

    r = pdk.Deck(
        layers=[geojson],
        initial_view_state=INITIAL_VIEW_STATE,
        tooltip=tooltip
    )

    st.pydeck_chart(r)
    progress_bar.progress(4/N_STEPS)


with st.spinner(text="Fetching batch of features used in the last run"):
    features_df = _load_batch_of_features_from_store(current_date)
    st.sidebar.write('✅ Inference features fetched from the store')
    progress_bar.progress(5/N_STEPS)


with st.spinner(text="Plotting time-series data"):
   
    row_indices = np.argsort(predictions_df['predicted_demand'].values)[::-1]
    n_to_plot = 10

    # plot each time-series with the prediction
    for row_id in row_indices[:n_to_plot]:
        fig = plot_one_sample(
            example_id=row_id,
            features=features_df,
            targets=predictions_df['predicted_demand'],
            predictions=pd.Series(predictions_df['predicted_demand'])
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)

    progress_bar.progress(6/N_STEPS)
