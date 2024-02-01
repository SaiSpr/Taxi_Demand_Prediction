import os
# from dotenv import load_dotenv

from src.paths import PARENT_DIR

# load key-value pairs from .env file located in the parent directory
# load_dotenv(PARENT_DIR / '.env')

# HOPSWORKS_API_KEY = 'DJmMdmfRSQgAWtbo.tTjG0ZJFhPy5MVXmMovoAmzpEbmcWzNvLJb9mHfeKBTJkhflD8YdgoH3E6CwpNEE'

# HOPSWORKS_PROJECT_NAME = 'uber_demand'


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
