🚖 Taxi Demand Predictor Project 🚀

Explore the fascinating world of predicting NYC taxi demand with this end-to-end time series ML project! Predict the next hour's ride count, helping taxi companies bridge the supply-demand gap and serve users better. We've sourced data from the NYC government's "TLC Trip Record Data" from January 2022 to June 2023.

📊 Sample Data:
Take a sneak peek at the dataset, with a focus on 'tpep_pickup_datetime' and 'PULocationID'. These columns are pivotal for forecasting ride numbers. We've even crafted a 'number_of_rides' column through smart grouping.

📈 Methodology:

    Preprocessing: Introduce new time slots missing in the original data.
    Modeling: Experiment with baseline estimators, XGBoost, and LightGBM. Guess what? LightGBM ruled the roost!
    Featurestore Magic: Leverage Hopsworks Featurestore to house features and models.
    GitHub Journey: Find the code on GitHub.
    Interactive Frontend: Dive into the Streamlit-powered app, featuring NYC's map with dark green zones highlighting high ride predictions, including the next ten hours.

🚀 Demo:
Experience the project live at this link. Hop on board and explore the world of taxi demand prediction! 🌟🗽🚕
