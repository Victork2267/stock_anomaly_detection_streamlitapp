#streamlit mods
import streamlit as st
import numpy as np

#stock mods
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

#Anomaly prediction
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from mpl_toolkits.mplot3d import Axes3D
from bokeh.models import Toggle, BoxAnnotation

#tweets mods
import pandas as pd
import jsonpickle
import re

#Graphs
import bs4 as bs
import pickle
import requests
import os
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, LabelSet

#General Info
gen_info = pd.read_csv("S&P_key_info.csv")

#Functions
import Streamlit_Stock_Sentiment_func as sss

outliers_fraction = 0.01

#Streamlit Starting line

#General Information Section
st.title("Stock price comparsion with twitter and News sentiments of S&P 500 companies")
st.sidebar.subheader("Configure parameters:")
option_SP = st.sidebar.selectbox("Select a S&P 500 company: ", gen_info["Company"].values)
start = st.sidebar.date_input("Select a Start date: ",dt.date(2015,11,1))
end = st.sidebar.date_input("Select an End date: ",dt.date(2019,11,1))
st.header(option_SP)
st.subheader("General Information on {}".format(option_SP))
st.write("headquarters: {}".format(str(gen_info["HeadQuarters"][gen_info["Company"] == option_SP]).split('Name')[0]))
st.write("Sector: {}".format(str(gen_info["Sector"][gen_info["Company"] == option_SP]).split('Name')[0]))
st.write("Sub-Industry: {}".format(str(gen_info["Sub_industry"][gen_info["Company"] == option_SP]).split('Name')[0]))
st.write("Founded: {}".format(str(gen_info["Founded"][gen_info["Company"] == option_SP]).split('Name')[0]))

#Stock section (Stock market does not open in the weekends, need to figure out a way to get the last week data from today)
st.subheader("Stock trend for the past 1 month")
today = dt.date.today().weekday()
ticker_select = []

for i in gen_info["Tickers"].loc[gen_info['Company'] == option_SP]:
    ticker_select.append(i)

#ticker_select = str(gen_info["Tickers"][gen_info["Company"] == option_SP]).split('Name')[0]
stock_df, status = sss.get_stock_data(ticker = ticker_select[0] , start=start , end=end)
stock_df["Date"] = stock_df.index.tolist()

#Get saved model
filename = r"ano_models/{}_ano_model.sav".format(ticker_select[0])
model_IF = pickle.load(open(filename, 'rb'))

#Download past data for anomaly detection
stock_df = sss.IF_prediction(df=stock_df,model=model_IF)

a = stock_df.loc[stock_df['anomaly_IF'] == -1, ['Volume', 'Close']]

largest_date_a = a[a['Volume']==a['Volume'].max()].reset_index()["Date"][0]
lowest_date_a = a[a['Volume']==a['Volume'].min()].reset_index()["Date"][0]

lar_date_5_for,lar_date_5_bac,low_date_5_for,low_date_5_bac = sss.get_date_range(largest_date=largest_date_a,lowest_date=lowest_date_a)

st.write(stock_df.head())

def datetime(x):
    return np.array(x, dtype=np.datetime64)

p1 = figure(x_axis_type="datetime", title="Stock prices on {}".format(option_SP))
p1.grid.grid_line_alpha=0.3
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Price'
p1.background_fill_color = "#dddddd"
p1.grid.grid_line_color = "white"

p1.line(datetime(stock_df.index), stock_df['High'], color='#A6CEE3', legend='High')
p1.line(datetime(stock_df.index), stock_df['Low'], color='#B2DF8A', legend='Low')
p1.line(datetime(stock_df.index), stock_df['Open'], color='#33A02C', legend='Open')
p1.line(datetime(stock_df.index), stock_df['Adj Close'], color='#FB9A99', legend='Adj Close')

p1.legend.location = "top_left"

ticker_pre_avg = np.array(stock_df['Adj Close'])
ticker_dates = np.array(stock_df.index, dtype=np.datetime64)

window_size = 30
window = np.ones(window_size)/float(window_size)
ticker_avg = np.convolve(ticker_pre_avg, window, 'same')

p11 = figure(x_axis_type="datetime", title="{} - One-Month Average".format(option_SP))
p11.grid.grid_line_alpha = 0
p11.xaxis.axis_label = 'Date'
p11.yaxis.axis_label = 'Price'
p11.ygrid.band_fill_color = "olive"
p11.ygrid.band_fill_alpha = 0.1

p11.circle(ticker_dates, ticker_pre_avg, size=4, legend='close',
          color='darkgrey', alpha=0.2)

p11.line(ticker_dates, ticker_avg, legend='avg', color='navy')
p11.legend.location = "top_left"

# anomaly prediction
a = stock_df.loc[stock_df['anomaly_IF'] == -1, ['Volume','High', 'Low']]

p12 = figure(x_axis_type="datetime", title="Anomaly detection on volume data - {}".format(option_SP))
p12.grid.grid_line_alpha=0.3
p12.xaxis.axis_label = 'Date'
p12.yaxis.axis_label = 'Volume'
p12.background_fill_color = "#dddddd"
p12.grid.grid_line_color = "white"

#shade +-3 day period of anomaly
green_box = BoxAnnotation(left=lar_date_5_bac, right=lar_date_5_for, fill_color='green', fill_alpha=0.1)
p12.add_layout(green_box)

red_box = BoxAnnotation(left=low_date_5_bac, right=low_date_5_for, fill_color='red', fill_alpha=0.1)
p12.add_layout(red_box)

p12.line(datetime(stock_df.index), stock_df['Volume'], color='#A6CEE3', legend='High')
p12.scatter(datetime(a.index), a['Volume'], color='navy', legend='Anomalies')

p12.legend.location = "top_left"

st.bokeh_chart(p1)
st.bokeh_chart(p11)
st.bokeh_chart(p12)

with st.spinner("Loading Corelation Data"):
    #Get data from yahoo
    sss.get_data_from_yahoo_1(start=start,end=end)

    #complie them into a correlation format
    sss.compile_data()

    #Get the correlation data
    df_corr_pre = pd.read_csv("sp500_joined_closes.csv")
    df_corr_new_ticker = df_corr_pre.corr()[str(ticker_select[0])]

    #Extract the top 5 and lowest 5 correlation values in relation to the selected ticker
    top_5_correlation_tickers_pos = df_corr_new_ticker.nlargest(6)[1:6]
    top_5_correlation_tickers_neg = df_corr_new_ticker.nsmallest(6)[1:6]

    #classify them into positive and negative correlations
    pos_list = top_5_correlation_tickers_pos.keys().tolist()
    neg_list = top_5_correlation_tickers_neg.keys().tolist()

    #Get the company name from ticker
    pos_list_com = []
    neg_list_com = []

    #pos_list_com_final = []
    #neg_list_com_final = []

    for ticker in pos_list:
        pos_list_com.append(str(gen_info["Company"].loc[gen_info['Tickers'] == ticker]).split("Name")[0])

    for ticker in neg_list:
        neg_list_com.append(str(gen_info["Company"].loc[gen_info['Tickers'] == ticker]).split("Name")[0])

    #Creating a dataframe for the positive and negative
    dict_pos_neg = {"Top 5 positively correlated S&P companies to {}".format(option_SP):pos_list_com,"Top 5 negatively correlated S&P companies to {}".format(option_SP):neg_list_com}
    df_pos_neg_show = pd.DataFrame(dict_pos_neg)

    st.subheader("Top 5 postively and negatively correlated companies with {}".format(option_SP))
    st.write(df_pos_neg_show)

    option_ocv_data = st.selectbox("Select a ocv data to display: ", np.asarray(stock_df.columns.tolist()))

    #Extracting their relevant ticker data
    df_1_pos, status_1 = sss.get_stock_data(ticker = pos_list[0] , start=start , end=end)
    df_2_pos, status_2 = sss.get_stock_data(ticker = pos_list[1] , start=start , end=end)
    df_3_pos, status_3 = sss.get_stock_data(ticker = pos_list[2] , start=start , end=end)
    df_4_pos, status_4 = sss.get_stock_data(ticker = pos_list[3] , start=start , end=end)
    df_5_pos, status_5 = sss.get_stock_data(ticker = pos_list[4] , start=start , end=end)

    df_1_neg, status_6 = sss.get_stock_data(ticker = neg_list[0] , start=start , end=end)
    df_2_neg, status_7 = sss.get_stock_data(ticker = neg_list[1] , start=start , end=end)
    df_3_neg, status_8 = sss.get_stock_data(ticker = neg_list[2] , start=start , end=end)
    df_4_neg, status_9 = sss.get_stock_data(ticker = neg_list[3] , start=start , end=end)
    df_5_neg, status_10 = sss.get_stock_data(ticker = neg_list[4] , start=start , end=end)

    # Graph it out (pos)
    p2 = figure(x_axis_type="datetime", title="Positively Corelated Companies on {} for {} data".format(option_SP,option_ocv_data))
    p2.grid.grid_line_alpha=0.3
    p2.xaxis.axis_label = 'Date'
    p2.yaxis.axis_label = 'Price'

    p2.line(datetime(stock_df.index), stock_df[option_ocv_data], color='#A6CEE3', legend=option_SP)
    p2.line(datetime(stock_df.index), df_1_pos[option_ocv_data], color='#B2DF8A', legend=str(pos_list_com[0]))
    p2.line(datetime(stock_df.index), df_2_pos[option_ocv_data], color='#33A02C', legend=str(pos_list_com[1]))
    p2.line(datetime(stock_df.index), df_3_pos[option_ocv_data], color='#FB9A99', legend=str(pos_list_com[2]))
    p2.line(datetime(stock_df.index), df_4_pos[option_ocv_data], color='#FF00FF', legend=str(pos_list_com[3]))
    p2.line(datetime(stock_df.index), df_5_pos[option_ocv_data], color='#7B68EE', legend=str(pos_list_com[4]))
    p2.legend.location = "top_left"

    st.bokeh_chart(p2)

    # Graph it out (neg)
    p3 = figure(x_axis_type="datetime", title="Negatively Corelated Companies on {} for {} data".format(option_SP,option_ocv_data))
    p3.grid.grid_line_alpha=0.3
    p3.xaxis.axis_label = 'Date'
    p3.yaxis.axis_label = 'Price'

    p3.line(datetime(stock_df.index), stock_df[option_ocv_data], color='#A6CEE3', legend=option_SP)
    p3.line(datetime(stock_df.index), df_1_neg[option_ocv_data], color='#B2DF8A', legend=str(neg_list_com[0]))
    p3.line(datetime(stock_df.index), df_2_neg[option_ocv_data], color='#33A02C', legend=str(neg_list_com[1]))
    p3.line(datetime(stock_df.index), df_3_neg[option_ocv_data], color='#FB9A99', legend=str(neg_list_com[2]))
    p3.line(datetime(stock_df.index), df_4_neg[option_ocv_data], color='#FF00FF', legend=str(neg_list_com[3]))
    p3.line(datetime(stock_df.index), df_5_neg[option_ocv_data], color='#7B68EE', legend=str(neg_list_com[4]))
    p3.legend.location = "top_left"

    st.bokeh_chart(p3)