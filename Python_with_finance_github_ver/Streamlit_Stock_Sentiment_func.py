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
import tweepy
import jsonpickle
import re
from textblob import TextBlob

#News mods
from newsapi import NewsApiClient

#Graphs
import bs4 as bs
import pickle
import requests
import os
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, LabelSet
#Functions
#Stocks

start = dt.date(2019,11,1)
end = dt.date(2019,11,5)

def get_stock_data(ticker, start, end):
    df_old = web.DataReader(ticker,"yahoo", start, end)
    df = df_old.dropna()
    loss_data = len(df_old) - len(df)
    if loss_data == 0:
        loss_status = "No NAs"
    elif loss_data > 0:
        loss_status = "No. of NAs: " + str(loss_data)

    return df, loss_status

# Creating IsolationForest model

outliers_fraction = 0.01

def create_model(df_past):
    data = df_past[["Volume","Close"]]
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(data)
    data = pd.DataFrame(np_scaled)
    model = IsolationForest(contamination=outliers_fraction)
    model.fit(data)
    return model

def IF_prediction(df,model):
    model = model
    data = df[["Volume","Close"]]
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(data)
    data = pd.DataFrame(np_scaled)
    df["anomaly_IF"] = list(model.predict(data))
    return df

def get_date_range(largest_date,lowest_date):
    lar_date_5_for = largest_date  + dt.timedelta(5)
    lar_date_5_bac = largest_date  - dt.timedelta(5)

    low_date_5_for = lowest_date  + dt.timedelta(5)
    low_date_5_bac = lowest_date  - dt.timedelta(5)
    return lar_date_5_for,lar_date_5_bac,low_date_5_for,low_date_5_bac


def save_sp500_tickers_1():
    resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = bs.BeautifulSoup(resp.text)
    table = soup.find("table",{'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    ticker_news = []

    for ticker in tickers:
        ticker_new = ticker.replace('\n','')
        ticker_news.append(ticker_new)

    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(ticker_news,f)

    return ticker_news

# save_sp500_tickers()
def get_data_from_yahoo_1(start,end,reload_sp500 = False):
    if reload_sp500:
        tickers = save_sp500_tickers_1()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = start
    end = end

    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            try:
                df = web.DataReader(ticker, 'yahoo', start, end)
                df.reset_index(inplace=True)
                df.set_index("Date", inplace=True)
                #df = df.drop("Symbol", axis=1)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
                print("Ticker: {} done".format(ticker))
            except:
                print("Ticker: {} not found".format(ticker))
        else:
            print('Already have {}'.format(ticker))

def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        try:
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)

            df.rename(columns={'Adj Close': ticker}, inplace=True)
            df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')

            if count % 10 == 0:
                print(count)
        except:
            print("ticker {} not present".format(ticker))

    main_df.to_csv("sp500_joined_closes.csv")
