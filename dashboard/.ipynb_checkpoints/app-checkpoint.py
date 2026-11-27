
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
 

st.set_page_config(
    page_title="Electricity Consumption Forecasting",
    page_icon="⚡",
    layout="wide"
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "daily_consumption.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "model_features.pkl"


@st.cache_data
def load_data():
    data = pd.read_csv(
        DATA_PATH,
        index_col=0,
        parse_dates=True
    )

    data.index.name = "datetime"
    data.sort_index(inplace=True)

    return data


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)

    return model, features


try:
    data = load_data()
    model, model_features = load_model()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()


st.sidebar.title("⚡ Electricity Forecasting")

st.sidebar.write(
    "Analyze household electricity consumption "
    "and predict future usage."
)

forecast_days = st.sidebar.selectbox(
    "Select forecasting period",
    [7, 30]
)

historical_days = st.sidebar.slider(
    "Historical data to display",
    min_value=30,
    max_value=365,
    value=90
)



st.title("⚡ Electricity Consumption Dashboard")

st.write(
    "Household Electricity Consumption Analysis "
    "and Machine Learning Forecasting"
)

st.caption(
    "Forecasting model: Optimized Random Forest Regression"
)

st.divider()



latest_consumption = data["Global_active_power"].iloc[-1]

average_consumption = data[
    "Global_active_power"
].mean()

maximum_consumption = data[
    "Global_active_power"
].max()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Latest Daily Consumption",
    f"{latest_consumption:.2f} kWh"
)

col2.metric(
    "Average Daily Consumption",
    f"{average_consumption:.2f} kWh"
)

col3.metric(
    "Maximum Daily Consumption",
    f"{maximum_consumption:.2f} kWh"
)


st.subheader("Historical Electricity Consumption")

historical_data = data.tail(historical_days)

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    historical_data.index,
    historical_data["Global_active_power"],
    linewidth=1.5
)

ax.set_xlabel("Date")
ax.set_ylabel("Consumption (kWh)")
ax.set_title("Daily Electricity Consumption")
ax.grid(alpha=0.3)

fig.tight_layout()

st.pyplot(fig)
plt.close(fig)



def generate_forecast(model, history_data, days, features):

    history = list(
        history_data["Global_active_power"].values
    )

    last_date = history_data.index[-1]

    forecasts = []

    for i in range(1, days + 1):

        future_date = last_date + pd.Timedelta(days=i)

        feature_values = {
            "day_of_week": future_date.dayofweek,
            "month": future_date.month,
            "day_of_year": future_date.dayofyear,
            "lag_1": history[-1],
            "lag_7": history[-7:][0],
            "rolling_mean_7": np.mean(history[-7:])
        }

        input_data = pd.DataFrame(
            [feature_values],
            columns=features
        )

        prediction = model.predict(input_data)[0]

        prediction = max(0, prediction)

        forecasts.append({
            "Date": future_date,
            "Predicted Consumption (kWh)": prediction
        })

        history.append(prediction)

    return pd.DataFrame(forecasts).set_index("Date")


st.divider()

st.subheader(
    f"Future {forecast_days}-Day Electricity Forecast"
)

if st.button("Generate Forecast", type="primary"):

    if len(data) < 7:
        st.error("At least seven historical daily records are required.")

    else:

        forecast = generate_forecast(
            model,
            data,
            forecast_days,
            model_features
        )

        st.session_state["forecast"] = forecast


if "forecast" in st.session_state:

    forecast = st.session_state["forecast"]

    if len(forecast) != forecast_days:
        st.info("Generate a new forecast for the selected period.")

    else:

        total_forecast = forecast[
            "Predicted Consumption (kWh)"
        ].sum()

        average_forecast = forecast[
            "Predicted Consumption (kWh)"
        ].mean()

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Predicted Consumption",
            f"{total_forecast:.2f} kWh"
        )

        col2.metric(
            "Average Daily Forecast",
            f"{average_forecast:.2f} kWh"
        )

        # Forecast chart
        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(
            data.index[-30:],
            data["Global_active_power"].tail(30),
            label="Historical Consumption"
        )

        ax.plot(
            forecast.index,
            forecast["Predicted Consumption (kWh)"],
            marker="o",
            linestyle="--",
            label="Predicted Consumption"
        )

        ax.set_title(
            f"{forecast_days}-Day Electricity Consumption Forecast"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Consumption (kWh)")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Forecast Results")

        st.dataframe(
            forecast.style.format("{:.2f}"),
            use_container_width=True
        )

        csv = forecast.to_csv().encode("utf-8")

        st.download_button(
            label="Download Forecast CSV",
            data=csv,
            file_name=f"{forecast_days}_day_forecast.csv",
            mime="text/csv"
        )



st.divider()

st.caption(
    "Electricity Consumption Analysis and Forecasting | "
    "Machine Learning Project"
)
