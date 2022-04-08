import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import time

url="https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"

st.header("Exploratory Data Analysis, Prediction and Timeseries forecasting of Omicron in India")
st.write("-------")

sidebar_options =  ["Introduction","Dataset","Visualization","Model Prediction","Timeseries Forecasting"]    
radio_options = ["Daily Cases","Correlation","Timestamp","Chloropleth"]

def load_data():
    df=pd.read_csv(url)
    df.drop(["iso_code","continent" ],1,inplace=True)
    return df

def processing():
    df=pd.read_csv(url)
    df = df[df["location"]=="India"]
    df.drop(["iso_code","continent" ],1,inplace=True)
    return df
    
def cmap1():
    statewise = pd.read_csv("https://raw.githubusercontent.com/Dharani-Sri/Omicron-Prediction/main/src/statewise_cases.csv")
    fig = px.choropleth_mapbox(statewise,
    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
    featureidkey='properties.ST_NM',
    locations='State',
    color='Active_cases',
    range_color=(0,10000),
    center={"lat": 21.7679, "lon": 78.8718},
    color_continuous_scale='Viridis',
    mapbox_style="carto-positron",
    zoom=3,height=700,
    title='Active cases of Indian states')
    
    fig.update_geos(fitbounds="locations", visible=False)
    return fig

def cmap2():
    statewise = pd.read_csv("https://raw.githubusercontent.com/Dharani-Sri/Omicron-Prediction/main/src/statewise_cases.csv")
    fig = px.choropleth_mapbox(statewise,
    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
    featureidkey='properties.ST_NM',
    locations='State',
    color='Cured_cases',
    range_color=(0,50000),
    center={"lat": 21.7679, "lon": 78.8718},
    mapbox_style="carto-positron",
    zoom=3,height=700,
    title='Recovered cases of Indian states')

    fig.update_geos(fitbounds="locations", visible=False)
    return fig

def prediction():
    df = pd.read_csv(url)
    df = df[df['date']>"2021-12-12"]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df.set_index('date', inplace=True)
    india_data = df[df["location"]=="India"]
    datewise_india = india_data.groupby(["location"]).agg({"total_cases":"sum","total_deaths":"sum","positive_rate":"sum"})
    return datewise_india

def main():
    df=pd.read_csv(url)
    st.sidebar.write("-----")
    st.sidebar.header("Explore our Project")
    mode=st.sidebar.selectbox("Please select from the following",sidebar_options)
    st.sidebar.write("-----")
    
    if mode == "Introduction":
        st.write("On November 26, 2021, the World Health Organization (WHO) classified a new variant, B.1.1.529, as a Variant of Concern and named it Omicron and on November 30, 2021, the United States also classified it as a Variant of Covid.\nCenters for Disease Control and Prevention is working with state and local public health officials to monitor the spread of Omicron. As of December 20, 2021, Omicron had been detected in every U.S. state and territory and continues to be the dominant variant in the United States.")    
        st.write("The Omicron variant spreads more easily than earlier variants of the virus that cause COVID-19, including the Delta variant. CDC expects that anyone with Omicron infection, regardless of vaccination status or whether or not they have symptoms, can spread the virus to others.")
        st.write("**Our goal is to understand the outbreak of OMICRON in India using Machine Learning Techniques.**" )
    
        
    if mode == "Dataset":
        st.subheader("Let us explore the dataset")
        st.write("It is necessary to work on collected data, pre-process them in order to obtain a consistent dataset and then extract the most relevant features. Here we can see the raw dataset....")
        if st.button("Load the dataset"):
            df = load_data()
            st.dataframe(df)
            time.sleep(10)
            
    if mode == "Model Prediction":
        st.subheader("Prediction chart over week")
        img=Image.open('img/prediction1.PNG')
        st.image(img)
        img=Image.open('img/prediction2.PNG')
        st.image(img)
        st.subheader("Model Prediction Table")
        model_predictions = prediction()       
        model_predictions = pd.read_csv('https://raw.githubusercontent.com/Dharani-Sri/Omicron-Prediction/main/src/file1.csv')
        st.dataframe(model_predictions)
        
    if mode == "Timeseries Forecasting":
        country = "India"
        filter_case = 'new_cases' 
        st.subheader("Timeseries Graph")
        st.write('Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.Here we can see the predictions made by the Prophet model.')
        st.image('img/forecast.png')
        df = df[df['location']==country]
        df.rename(columns={"date": "ds", filter_case: "y"},inplace=True) 
        df['ds'] = pd.to_datetime(df['ds'],infer_datetime_format=True)
        df = df[df['ds']>"2020-12-12"]
        df['y'] = df['y'].astype(float)
        df = df[['y','ds']]
        
    if mode == "Visualization":
        choice = st.sidebar.radio("Choose your charts",radio_options)        
        df=processing()
        df = df[df['date']>"2021-12-12"]
        omicron = df[['location', 'date', 'total_cases', 'new_cases','new_cases_smoothed', 'total_deaths', 'new_deaths',
       'new_deaths_smoothed', 'total_cases_per_million','new_cases_per_million', 'new_cases_smoothed_per_million',
       'total_deaths_per_million', 'new_deaths_per_million']]
        
        
        if choice == "Daily Cases":
            st.subheader("Here we see the Intractive visualization of past Data")
            st.write("Data Visualization is the first step towards getting an insight into a large data set in every project. Once the data has been acquired and preprocessed , the next step is Exploratory Data Analysis which kicks off with visualization of the data. The aim here is to extract useful information from the data.")
            fig1 = px.line(omicron,x='date',y='total_cases',title='Daily Confirmed Omicron Cases')
            st.plotly_chart(fig1)
            fig2 = px.line(omicron,x='date',y='total_deaths',title='Daily Confirmed death Cases',color_discrete_sequence = ["red"])
            st.plotly_chart(fig2)
           
        if choice == "Correlation":
            st.subheader("Let us check the correlation of the columns")
            plt.figure(figsize=(20,15))
            hm = sns.heatmap(omicron.corr(),annot=True)
            st.pyplot(hm.figure)
            
        if choice == "Timestamp":
            st.subheader("Daily progress of New Cases")
            st.image("img/progress.PNG")
            
        if choice == "Chloropleth":
            fig1 = cmap1()
            st.plotly_chart(fig1)
            fig2 = cmap2()
            st.plotly_chart(fig2)     
       
main()
