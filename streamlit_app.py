import streamlit as st
import pandas as pd
import plotly.express as px
import calendar
from datetime import datetime

# --- Function to create sample data (since no actual data loading is happening) ---


def create_sample_data():
    data = {
        'Date': pd.to_datetime(['2024-01-05', '2024-01-12', '2024-01-19', '2024-01-26', '2024-02-02', '2024-02-09',
                               '2024-02-16', '2024-02-23', '2024-03-01', '2024-03-08', '2024-03-15', '2024-03-22']),
        'Distance (km)': [2.5, 3.1, 2.8, 3.5, 2.9, 3.3, 3.0, 2.7, 3.6, 3.2, 2.4, 1.9],
        'Minutes Actives': [45, 55, 50, 60, 48, 58, 52, 47, 62, 53, 40, 23],
        'Points Cardio': [60, 75, 68, 82, 65, 78, 70, 63, 85, 72, 55, 36],
        'Pas Total': [6000, 7200, 6500, 7800, 6800, 7500, 7100, 6200, 8000, 7300, 5892, 2100],
        "Durée de l'activité \"Marche à pied\" (ms)": [0] * 12,
        "Durée de l'activité \"Vélo\" (ms)": [0] * 12,
        "Durée de l'activité \"Course à pied\" (ms)": [0] * 12

    }
    df = pd.DataFrame(data)
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['week'] = df['Date'].dt.isocalendar().week
    return df

# --- Check if required columns exist; otherwise, create them with default values ---


def check_and_create_columns(df):
    required_columns = {
        "Durée de l'activité \"Marche à pied\" (ms)": 0,
        "Durée de l'activité \"Vélo\" (ms)": 0,
        "Durée de l'activité \"Course à pied\" (ms)": 0,
    }
    for col, default_value in required_columns.items():
        if col not in df.columns:
            st.warning(
                f"Colonne '{col}' non trouvée. Initialisée à {default_value}.")
            df[col] = default_value
    return df


# --- Create sample data ---
df = create_sample_data()
df = check_and_create_columns(df)  # Ensure necessary columns exist

# --- Sidebar for Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisissez une page", [
                        "Vue d'ensemble", "Tendances", "Analyses", "Carte"])

# --- Main Dashboard ---
st.title("Tableau de Bord Fitness")

if page == "Vue d'ensemble":
    st.header("Vue d'ensemble")

    # --- Calendar Heatmap ---
    st.subheader("Calendrier d'Activités")

    # Get min and max year from data
    min_year = df['year'].min()
    max_year = df['year'].max()

    # Year selection
    selected_year = st.selectbox("Sélectionnez l'année", range(
        min_year, max_year + 1), index=len(range(min_year, max_year + 1))-1)  # select current year

    # Filter data for the selected year
    df_year = df[df['year'] == selected_year]

    # Create a dictionary to hold the data for the heatmap
    heatmap_data = {}
    for _, row in df_year.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d')
        heatmap_data[date_str] = row['Distance (km)']

    # Display calendar for the selected year
    for month_num in range(1, 13):
        cal_str = calendar.month(selected_year, month_num)

        # Create a list of days with activity
        days_with_activity = []
        for date, value in heatmap_data.items():
            dt_obj = datetime.strptime(date, '%Y-%m-%d')
            if dt_obj.year == selected_year and dt_obj.month == month_num:
                days_with_activity.append(dt_obj.day)

        # Replace days with activity with a colored marker
        for day in days_with_activity:
            cal_str = cal_str.replace(
                f"{day:2d}", f"**[{day:2d}]**")  # Bold the number

        st.markdown(f"**{calendar.month_name[month_num]} {selected_year}**")
        st.text(cal_str)

    # --- Key Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Distance Totale (km)",
                f"{df_year['Distance (km)'].sum():.1f}")
    col2.metric("Minutes Actives", int(df_year['Minutes Actives'].sum()))
    col3.metric("Points Cardio", int(df_year['Points Cardio'].sum()))
    col4.metric("Pas Total", int(df_year['Pas Total'].sum()))

    # --- Recent Activity Summary ---
    st.subheader("Résumé des Activités Récentes")
    st.dataframe(df_year.sort_values(by='Date', ascending=False))

elif page == "Tendances":
    st.header("Tendances")

    # --- Time Series Charts ---
    st.subheader("Distance au fil du temps")
    fig_distance = px.line(df, x='Date', y='Distance (km)',
                           title='Distance au fil du temps')
    st.plotly_chart(fig_distance)

    st.subheader("Minutes Actives au fil du temps")
    fig_minutes = px.line(df, x='Date', y='Minutes Actives',
                          title='Minutes Actives au fil du temps')
    st.plotly_chart(fig_minutes)

    st.subheader("Points Cardio au fil du temps")
    fig_cardio = px.line(df, x='Date', y='Points Cardio',
                         title='Points Cardio au fil du temps')
    st.plotly_chart(fig_cardio)

    st.subheader("Pas Total au fil du temps")
    fig_steps = px.line(df, x='Date', y='Pas Total',
                        title='Pas Total au fil du temps')
    st.plotly_chart(fig_steps)

    # --- Select box for choosing the metric to plot ---
    selected_metric = st.selectbox("Choisissez une métrique", [
                                   'Distance (km)', 'Minutes Actives', 'Points Cardio', 'Pas Total'])

    # --- Bar chart by week ---
    st.subheader(f"{selected_metric} par semaine")
    df_weekly = df.groupby(['year', 'week'])[
        selected_metric].sum().reset_index()
    fig_weekly = px.bar(df_weekly, x='week', y=selected_metric, color='year', title=f'{selected_metric} par semaine',
                        labels={'week': 'Semaine', selected_metric: selected_metric})
    st.plotly_chart(fig_weekly)

    # --- Bar chart by month ---
    st.subheader(f"{selected_metric} par mois")
    df_monthly = df.groupby(['year', 'month'])[
        selected_metric].sum().reset_index()
    fig_monthly = px.bar(df_monthly, x='month', y=selected_metric, color='year', title=f'{selected_metric} par mois',
                         labels={'month': 'Mois', selected_metric: selected_metric})
    st.plotly_chart(fig_monthly)


elif page == "Analyses":
    st.header("Analyses")

    # --- Histograms ---
    st.subheader("Distribution de la Distance (km)")
    fig_hist_distance = px.histogram(
        df, x='Distance (km)', nbins=5, title='Distribution de la Distance (km)')
    st.plotly_chart(fig_hist_distance)

    st.subheader("Distribution des Minutes Actives")
    fig_hist_minutes = px.histogram(
        df, x='Minutes Actives', nbins=5, title='Distribution des Minutes Actives')
    st.plotly_chart(fig_hist_minutes)

    st.subheader("Distribution des Points Cardio")
    fig_hist_cardio = px.histogram(
        df, x='Points Cardio', nbins=5, title='Distribution des Points Cardio')
    st.plotly_chart(fig_hist_cardio)

    st.subheader("Distribution des Pas Total")
    fig_hist_steps = px.histogram(
        df, x='Pas Total', nbins=5, title='Distribution des Pas Total')
    st.plotly_chart(fig_hist_steps)

    # --- Scatter plots ---
    st.subheader("Relation entre Distance et Minutes Actives")
    fig_scatter = px.scatter(df, x='Distance (km)', y='Minutes Actives',
                             title='Relation entre Distance et Minutes Actives')
    st.plotly_chart(fig_scatter)

    st.subheader("Relation entre Distance et Points Cardio")
    fig_scatter2 = px.scatter(df, x='Distance (km)', y='Points Cardio',
                              title='Relation entre Distance et Points Cardio')
    st.plotly_chart(fig_scatter2)


elif page == "Carte":
    st.header("Carte")
    st.write("Fonctionnalité de carte non implémentée dans cet exemple (nécessite des données de localisation).")
    # If you had latitude and longitude data, you would use st.map() here.
