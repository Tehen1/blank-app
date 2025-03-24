import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px

# Sample data
data = {
    'timestamp': pd.date_range(start='2025-03-24', periods=100, freq='T'),
    'latitude': np.random.uniform(low=40.0, high=41.0, size=100),
    'longitude': np.random.uniform(low=-74.0, high=-73.0, size=100),
    'speed': np.random.uniform(low=0, high=30, size=100)
}

df = pd.DataFrame(data)

# Dashboard Title
st.title('Bike Tracking Dashboard')

# Display DataFrame
st.subheader('Bike Data')
st.dataframe(df)

# Plotly Map
fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', hover_name='timestamp',
                        color='speed', size='speed', zoom=10, height=300)
fig.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig)

# Speed over Time Chart
st.subheader('Speed over Time')
speed_fig = px.line(df, x='timestamp', y='speed', title='Speed over Time')
st.plotly_chart(speed_fig)

# Update Data Example
st.subheader('Live Data Update')
live_data = st.empty()

for i in range(10):
    time.sleep(1)
    new_data = {
        'timestamp': pd.date_range(start='2025-03-24', periods=1, freq='T'),
        'latitude': [np.random.uniform(low=40.0, high=41.0)],
        'longitude': [np.random.uniform(low=-74.0, high=-73.0)],
        'speed': [np.random.uniform(low=0, high=30)]
    }
    new_df = pd.DataFrame(new_data)
    df = pd.concat([df, new_df], ignore_index=True)
    live_data.dataframe(df.tail(10))

st.button("Refresh Data")
