# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 23:14:35 2021

@author: chibi
"""

# TODO
# 会社情報がでてこない
# 箱ひげ図のホバーで表示する値は元の値にする。

#! pip install yahoo_fin
import yahoo_fin.stock_info as si

# import fix_yahoo_finance as yf
import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from plotly.subplots import make_subplots
import sys
import plotly.express as px
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import numpy as np
import dash

# import dash_core_components as dcc
# import dash_html_components as html
from dash import dcc, html
import plotly.graph_objs as go

# %matplotlib inline
from dash.dependencies import Input, Output, State

# In[]:

stock_table = pd.read_csv("sbi_tickers.csv")
stock_list = list(pd.read_csv("sbi_tickers.csv")["symbol"])
stock_business = list(pd.read_csv("sbi_tickers.csv")["Business Content"])

stock_dict_list = []
for stk in stock_list:
    stock_dict = dict(zip(["label", "value"], [stk, stk]))
    stock_dict_list.append(stock_dict)


def getQuarterEPS(ticker):
    obj_ticker = yf.Ticker(ticker)
    q_eps = obj_ticker.earnings_dates
    q_eps = q_eps.dropna(how="all", axis=0)
    q_eps = q_eps.fillna(0)
    q_eps_copy = q_eps.copy()
    est, report, time = None, None, None
    for i in range(len(q_eps) - 1, -1, -1):
        est_new = q_eps_copy.iloc[i, 0]
        report_new = q_eps_copy.iloc[i, 1]
        time_new = q_eps_copy.index[i]

        if (est != est_new) and (report != report_new) and (time != time_new):
            est, report, time = est_new, report_new, time_new
            continue
        else:
            q_eps.drop(index=q_eps.index[i], inplace=True)

    q_eps["Earnings Date"] = [str(t)[:7] for t in q_eps.index]
    q_eps = q_eps.set_index("Earnings Date")
    return q_eps


def calcUpDownRate(slct_stock):
    plotdata = pd.DataFrame()
    prisedata = pd.DataFrame()
    for ticker in slct_stock:
        ticker_get_data = si.get_data(ticker)
        prisedata = pd.concat([prisedata, ticker_get_data["close"]], axis=1)
        ticker_get_data = ticker_get_data.sort_index(axis="index", ascending=False)
        ticker_get_data["diff"] = ticker_get_data["close"] - ticker_get_data["open"]
        ticker_get_data["up-down_day"] = ticker_get_data["close"] / ticker_get_data["open"]
        ticker_get_data["up-down_week"] = ticker_get_data["close"] / ticker_get_data["close"].shift(-5)
        ticker_get_data["up-down_month"] = ticker_get_data["close"] / ticker_get_data["close"].shift(-20)
        if len(ticker_get_data) < 240:
            ticker_get_data["up-down_year"] = ticker_get_data["close"] / ticker_get_data["close"].shift(
                -len(ticker_get_data) + 1
            )
        else:
            ticker_get_data["up-down_year"] = ticker_get_data["close"] / ticker_get_data["close"].shift(-240)
        ticker_get_data["日変動率"] = (ticker_get_data["up-down_day"] - 1) * 100
        ticker_get_data["週変動率"] = (ticker_get_data["up-down_week"] - 1) * 100
        ticker_get_data["月変動率"] = (ticker_get_data["up-down_month"] - 1) * 100
        ticker_get_data["年変動率"] = (ticker_get_data["up-down_year"] - 1) * 100
        ticker_get_data = ticker_get_data.iloc[0]
        plotdata = pd.concat([plotdata, ticker_get_data], axis=1)
    plotdata = plotdata.transpose()
    prisedata.columns = slct_stock
    return plotdata


def macd(df):
    FastEMA_period = 12  # 短期EMAの期間
    SlowEMA_period = 26  # 長期EMAの期間
    SignalSMA_period = 9  # SMAを取る期間
    df["MACD"] = df["close"].ewm(span=FastEMA_period).mean() - df["close"].ewm(span=SlowEMA_period).mean()
    df["Signal"] = df["MACD"].rolling(SignalSMA_period).mean()
    return df


def rsi(df):
    # 前日との差分を計算
    df_diff = df["close"].diff(1)
    # 計算用のDataFrameを定義
    df_up, df_down = df_diff.copy(), df_diff.copy()
    # df_upはマイナス値を0に変換
    # df_downはプラス値を0に変換して正負反転
    df_up[df_up < 0] = 0
    df_down[df_down > 0] = 0
    df_down = df_down * -1
    # 期間14でそれぞれの平均を算出
    df_up_sma14 = df_up.rolling(window=14, center=False).mean()
    df_down_sma14 = df_down.rolling(window=14, center=False).mean()
    # RSIを算出
    df["RSI"] = 100.0 * (df_up_sma14 / (df_up_sma14 + df_down_sma14))
    return df


def calcAnalyticMetrics(ticker, period_ratio):
    ticker_price = si.get_data(ticker)
    ticker_price = ticker_price.iloc[-int(20 * period_ratio) :]
    ticker_price = ticker_price.drop("ticker", axis=1)
    ticker_price_init = ticker_price / ticker_price.iloc[0]
    ticker_price["datetime"] = pd.to_datetime(ticker_price.index, unit="ms")
    ticker_price_init["datetime"] = pd.to_datetime(ticker_price.index, unit="ms")

    # SMAを計算
    ticker_price["SMA5"] = ticker_price["close"].rolling(window=5).mean()
    ticker_price["SMA25"] = ticker_price["close"].rolling(window=25).mean()
    SMA5 = ticker_price["SMA5"].dropna()
    SMA25 = ticker_price["SMA25"].dropna()
    if len(ticker_price) > 5:
        ticker_price_init["SMA5"] = SMA5 / SMA5.iloc[0]
    else:
        ticker_price_init["SMA5"] = [None] * len(ticker_price_init)
    if len(ticker_price) > 25:
        ticker_price_init["SMA25"] = SMA25 / SMA25.iloc[0]
    else:
        ticker_price_init["SMA25"] = [None] * len(ticker_price_init)

    # MACDを計算する
    ticker_price = macd(ticker_price)
    # RSIを算出
    ticker_price = rsi(ticker_price)
    return ticker_price, ticker_price_init


def calcAnalyticMetricsList(ticker_compared, period_ratio):
    data_compared_list = []
    for ticker in ticker_compared:
        ticker_price = si.get_data(ticker)
        ticker_price = ticker_price.iloc[-int(20 * period_ratio) :]
        ticker_price = ticker_price.drop("ticker", axis=1)
        ticker_price_init = ticker_price / ticker_price.iloc[0]
        ticker_price_init["datetime"] = pd.to_datetime(ticker_price_init.index, unit="ms")
        data_compared_list.append(ticker_price_init)
    return data_compared_list


def createAnalyticMetricsChart(df, df_init, ticker_name, df_compared_list, ticker_compared):
    # figを定義
    fig = make_subplots(rows=4, cols=1, shared_xaxes=False, row_heights=[6, 2, 2, 2], x_title="Date")
    # Candlestick
    i = 0
    # 陽線のカラーリスト
    positive_colors = ["#1f77b4", "#17becf", "#08519c", "#006d2c"]
    # 陰線のカラーリスト
    negative_colors = ["#ff7f0e", "#e377c2", "#f4a582", "#d6604d"]

    if len(df_compared_list) > 0:
        for ticker_compared, df_compared in zip(ticker_compared, df_compared_list):
            fig.add_trace(
                go.Candlestick(
                    x=df_compared["datetime"],
                    open=df_compared["open"],
                    high=df_compared["high"],
                    low=df_compared["low"],
                    close=df_compared["close"],
                    name=ticker_compared,
                    increasing_line_color=positive_colors[i],  # 陽線の色
                    decreasing_line_color=negative_colors[i],  # 陰線の色
                ),
                row=1,
                col=1,
            )
            i += 1

    fig.add_trace(
        go.Candlestick(
            x=df_init["datetime"],
            open=df_init["open"],
            high=df_init["high"],
            low=df_init["low"],
            close=df_init["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )
    # SMA
    fig.add_trace(
        go.Scatter(
            x=df_init["datetime"],
            y=df_init["SMA5"],
            name="SMA5",
            mode="lines",
            hovertemplate="DATE:%{x}: <br>Prise($):%{y}",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_init["datetime"],
            y=df_init["SMA25"],
            name="SMA25",
            mode="lines",
            hovertemplate="DATE:%{x}: <br>Prise($):%{y}",
        ),
        row=1,
        col=1,
    )

    # Volume
    fig.add_trace(
        go.Bar(x=df["datetime"], y=df["volume"], name="Volume", hovertemplate="DATE:%{x}: <br>Volume:%{y}"),
        row=2,
        col=1,
    )

    # MACD
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["MACD"],
            name="MACD",
            mode="lines",
            hovertemplate="DATE:%{x}: <br>MACD:%{y}",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["Signal"],
            name="Signal",
            mode="lines",
            hovertemplate="DATE:%{x}: <br>Signal:%{y}",
        ),
        row=3,
        col=1,
    )

    # RSI
    fig.add_trace(
        go.Scatter(x=df["datetime"], y=df["RSI"], name="RSI", mode="lines", hovertemplate="DATE:%{x}: <br>RSI:%{y}"),
        row=4,
        col=1,
    )

    # Layout
    fig.update_layout(
        title={
            "text": "Technical Charts : {}".format(ticker_name),
            # "y":0.99,
            "x": 0.5,
            # "size":40
        }
    )

    fig.update_layout(
        title=dict(font=dict(size=25, family="Gravitas")),
        hoverlabel=dict(font=dict(size=20)),
        plot_bgcolor=fig_bgColor,
        paper_bgcolor=web_bgColor,
        legend=dict(
            traceorder="reversed",
            title_font_family="Times New Roman",
            font=dict(family="Courier", size=12, color="black"),
            bgcolor="ivory",
            bordercolor="Black",
            borderwidth=2,
        ),
    )

    # y軸名を定義
    fig.update_yaxes(title_text="株価", row=1, col=1)  # ,title_font_size=10,tickfont_size=10)
    fig.update_yaxes(title_text="出来高", row=2, col=1)  # ,title_font_size=10,tickfont_size=10)
    fig.update_yaxes(title_text="MACD", row=3, col=1)  # ,title_font_size=10,tickfont_size=10)
    fig.update_yaxes(title_text="RSI", row=4, col=1)  # ,title_font_size=10,tickfont_size=10)

    # 日付リストを取得
    d_all = pd.date_range(start=df["datetime"].iloc[0], end=df["datetime"].iloc[-1])
    start = df["datetime"].iloc[0]
    end = df["datetime"].iloc[-1]
    # 株価データの日付リストを取得
    d_obs = [d.strftime("%Y-%m-%d") for d in df["datetime"]]

    # 株価データの日付データに含まれていない日付を抽出
    d_breaks = [d for d in d_all.strftime("%Y-%m-%d").tolist() if not d in d_obs]

    for i in [1, 2, 3, 4]:
        fig.update_xaxes(
            range=[start, end], rangebreaks=[dict(values=d_breaks)], title_font_size=20, tickfont_size=15, row=i, col=1
        )
    fig.update(layout_xaxis_rangeslider_visible=False)
    return fig


def text2list(text):
    list_ = text.split(",")
    return list_


# %% HTML layout

main_color = "chocolate"
fig_bgColor = "bisque"
web_bgColor = "ivory"
value_bf = ""
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.layout = html.Div(
    [
        html.H1(
            "StockAnalizer KimChart",
            style={
                "textAlign": "center",
                "color": "snow",
                # 'fontSize': 30,
                "backgroundColor": main_color,
            },
        ),
        html.Div(
            [
                html.Div(children="Select target stocks...", style={"fontSize": 20}),
                dcc.Dropdown(
                    id="stock_select",
                    options=stock_dict_list,
                    multi=True,
                    # value=["AAPL", "AMZN", "META", "GOOG", "MSFT", "TSLA", "NVDA"],
                    value=["AAPL", "NVDA"],
                    # clearable=False
                    style={
                        "background-color": fig_bgColor,
                        "border-color": main_color,
                        "color": main_color,
                    },
                ),
                dcc.Input(
                    id="stock_select_text",
                    placeholder='Input tickers in the following text format "AAA,BBB,CCC,..."',
                    type="text",
                    value="",
                    style={
                        "fontSize": 20,
                        "background-color": fig_bgColor,
                        "border-color": main_color,
                        # "color":main_color,
                        "height": "30px",
                        "width": "90vw",
                        "margin-top": "10px",
                    },
                ),
            ]
        ),
        html.Button(
            "Get Stock Information",
            id="button",
            style={
                "fontSize": 20,
                "background-color": main_color,
                "color": web_bgColor,
                "height": "30px",
                "width": "300px",
                "margin-top": "10px",
                "margin-bottom": "20px",
                "margin-left": "10px",
            },
        ),
        html.Div(
            children="👇👇👇👇👇",
            style={
                "fontSize": 20,
                # 'height': 'vh',
            },
        ),
        html.Div(
            id="output-container-button",
            children="Enter a value and press submit",
            style={
                "fontSize": 20,
                "color": "snow",
                "backgroundColor": main_color,
            },
        ),
        # 左側
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(children="Select X-axis period...", style={"fontSize": 20}),
                                dcc.Dropdown(
                                    id="dwm_select",
                                    options=[
                                        {"label": "DAY", "value": "day"},
                                        {"label": "WEEK", "value": "week"},
                                        {"label": "MONTH", "value": "month"},
                                        {"label": "YEAR", "value": "year"},
                                    ],
                                    value="day",
                                    clearable=False,
                                    style={"background-color": fig_bgColor, "color": main_color},
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Div(children="Select Y-axis period...", style={"fontSize": 20}),
                                dcc.Dropdown(
                                    id="dwm_select2",
                                    options=[
                                        {"label": "DAY", "value": "day"},
                                        {"label": "WEEK", "value": "week"},
                                        {"label": "MONTH", "value": "month"},
                                        {"label": "YEAR", "value": "year"},
                                    ],
                                    value="week",
                                    clearable=False,
                                    style={"background-color": fig_bgColor, "color": main_color},
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                dcc.Graph(id="scatter_chart", style={"width": "50vw", "height": "70vh"}),
                            ]
                        ),
                        html.Div(
                            id="ticker_info1",
                            children="----",
                            style={"fontSize": 20, "backgroundColor": fig_bgColor, "height": "15vh"},
                        ),
                        html.Div(
                            id="ticker_info2",
                            children="----",
                            style={"fontSize": 20, "backgroundColor": fig_bgColor, "height": "5vh"},
                        ),
                        html.Div(
                            [
                                dcc.Graph(id="kessan_chart", style={"width": "50vw", "height": "50vh"}),
                                dcc.Graph(id="kessan_chart2", style={"width": "50vw", "height": "50vh"}),
                                dcc.Graph(id="kessan_chart3", style={"width": "50vw", "height": "50vh"}),
                            ]
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flex-direction": "column",
                        "height": "150vh",
                        "margin-top": "20px",
                    },
                ),
                # 右側
                html.Div(
                    [
                        dcc.RadioItems(
                            id="period_select",
                            options=[
                                {"label": "HalfMonth", "value": 0.5},
                                {"label": "Month", "value": 1},
                                {"label": "HalfYear", "value": 6},
                                {"label": "Year", "value": 12},
                                {"label": "5Years", "value": 60},
                            ],
                            value=1,
                            persistence=True,
                            style={"fontSize": 20, "margin-left": "50px", "margin-top": "20px"},
                            labelStyle={"display": "inline-block", "padding": "10px"},
                        ),
                        dcc.Dropdown(
                            id="ticker_compared",
                            options=stock_dict_list,
                            multi=True,
                            value=[],
                            style={
                                "border-color": main_color,
                                "color": main_color,
                                "width": "15vw",
                                "margin-left": "30px",
                            },
                        ),
                        dcc.Graph(
                            id="chart_d",
                            style={
                                "height": "240vh",
                                "width": "50vw",
                            },
                        ),
                    ]
                ),
            ],
            style={"display": "flex", "flex-direction": "row", "margin-top": "10px", "height": "250vh"},
        ),
    ],
    style={"backgroundColor": web_bgColor},  # 背景色
)

# %% CALLBACK


@app.callback(
    dash.dependencies.Output("scatter_chart", "figure"),
    [
        dash.dependencies.Input("dwm_select", "value"),
        dash.dependencies.Input("dwm_select2", "value"),
        [dash.dependencies.State("stock_select", "value")],
        [dash.dependencies.State("stock_select_text", "value")],
        [dash.dependencies.Input("button", "n_clicks")],
    ],
)
def update_graph(dwm1, dwm2, slct_stock, slct_stock_text, ncli):
    slct_stock = slct_stock[0]
    if slct_stock_text[0] != "":
        slct_stock = slct_stock_text[0].split(",")

    plotdata = calcUpDownRate(slct_stock)

    fig = go.Figure()
    for ticker, s in zip(plotdata["ticker"], plotdata["up-down_" + dwm2]):
        ticker_info = yf.Ticker(ticker)
        finance = ticker_info.quarterly_income_stmt
        basicEPS = finance.loc["Basic EPS"].iloc[0]
        if basicEPS < 0:
            basicEPS = 5

        x = plotdata[plotdata["ticker"] == ticker]["up-down_" + dwm1]
        y = plotdata[plotdata["ticker"] == ticker]["up-down_" + dwm2]
        fig.add_hline(y=1, line_dash="dash", line_color="red", opacity=0.6)
        fig.add_vline(x=1, line_dash="dash", line_color="red", opacity=0.6)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers+text",
                name=ticker,
                hovertemplate="BasicEPS: " + str(basicEPS),
                marker={
                    "size": basicEPS * 15,
                },
            )
        )

    fig.update_layout(
        title={
            "text": "Prise Up-Down for a specified period",
            "x": 0.5,
        }
    )

    fig.update_layout(
        title=dict(font=dict(size=25, family="Old Standard TT")),
        hoverlabel=dict(font=dict(size=20)),
        paper_bgcolor=web_bgColor,
        plot_bgcolor=fig_bgColor,
        legend=dict(
            traceorder="reversed",
            title_font_family="Times New Roman",
            font=dict(family="Courier", size=10, color="black"),
            bgcolor="ivory",
            bordercolor="Black",
            borderwidth=2,
        ),
    )
    fig.update_xaxes(title=dwm1 + " up-down", title_font_family="Open Sans", title_font_size=20, tickfont_size=20)
    fig.update_yaxes(title=dwm2 + " up-down", title_font_family="Open Sans", title_font_size=20, tickfont_size=20)
    # fig.show()
    return fig


@app.callback(
    dash.dependencies.Output("kessan_chart", "figure"),
    [dash.dependencies.State("stock_select", "value")],
    [dash.dependencies.State("stock_select_text", "value")],
    [Input("scatter_chart", "hoverData")],
)
def update_output(slct_stock, slct_stock_text, hoverData):
    if slct_stock_text != "":
        slct_stock = slct_stock_text.split(",")
    n = hoverData["points"][0]["curveNumber"]
    ticker = slct_stock[n]
    q_eps = getQuarterEPS(ticker)
    time_ = q_eps.index
    actual_eps = q_eps["Reported EPS"].astype(float)
    estimated_eps = q_eps["EPS Estimate"].astype(float)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=time_,
            y=actual_eps,
            name="Actual",
            marker_color="indianred",
            marker_line_color="maroon",
            marker_line_width=2.5,
            opacity=0.8,
            hovertemplate="EPS:%{y}",
            text=actual_eps,
        )
    )
    # fig.add_trace(px.bar(DATA_EarnHist, x='time', y='EPS Actual',title="EPS",opacity=0.5)
    fig.add_trace(
        go.Bar(
            x=time_,
            y=estimated_eps,
            name="Estimate",
            marker_color="lightsalmon",
            marker_line_color="tomato",
            marker_line_width=2.5,
            opacity=0.8,
            hovertemplate="EPS:%{y}",
            text=estimated_eps,
        )
    )

    fig.update_layout(
        title={
            "text": "EPS ({})".format(ticker),
            "x": 0.5,
        }
    )

    fig.update_layout(
        title=dict(font=dict(size=25, family="Gravitas")),
        plot_bgcolor=fig_bgColor,
        paper_bgcolor=web_bgColor,
        hoverlabel=dict(font=dict(size=20)),
    )
    fig.update_layout(barmode="group", xaxis=dict(tickmode="array", tickvals=time_, ticktext=time_, tickangle=-45))
    fig.update_yaxes(title="Profit per stock")
    # fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
    #               marker_line_width=1.5, opacity=0.6)
    return fig


@app.callback(
    dash.dependencies.Output("kessan_chart2", "figure"),
    [dash.dependencies.State("stock_select", "value")],
    [dash.dependencies.State("stock_select_text", "value")],
    [Input("scatter_chart", "hoverData")],
)
def update_output(slct_stock, slct_stock_text, hoverData):
    if slct_stock_text != "":
        slct_stock = slct_stock_text.split(",")
    n = hoverData["points"][0]["curveNumber"]
    ticker = slct_stock[n]
    ticker_info = yf.Ticker(ticker)
    q_finance = ticker_info.quarterly_income_stmt
    Q_Data = q_finance.loc[["Total Revenue", "Operating Income"], :].transpose()
    time_ = [str(t)[:7] for t in Q_Data.index]
    Q_Data["総売上"] = Q_Data["Total Revenue"].astype(float) / 10**6
    Q_Data["営業利益"] = Q_Data["Operating Income"].astype(float) / 10**6
    Q_Data["営業利益率"] = Q_Data["営業利益"] / Q_Data["総売上"] * 100

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=time_,
            y=Q_Data["総売上"],
            text=Q_Data["総売上"],
            name="総売上",
            yaxis="y1",
            # marker_color='gold',
            hovertemplate="総売上:%{y}",
            marker_color="gold",
            marker_line_color="goldenrod",
            marker_line_width=2.5,
            opacity=0.6,
        )
    )

    fig.add_trace(
        go.Bar(
            x=time_,
            y=Q_Data["営業利益"],
            text=Q_Data["営業利益"],
            name="営業利益",
            # marker_color='gold',
            hovertemplate="営業利益:%{y}",
            yaxis="y1",
            marker_color="darkseagreen",
            marker_line_color="olivedrab",
            marker_line_width=2.5,
            opacity=0.6,
        )
    )

    fig.add_trace(
        go.Line(
            x=time_,
            y=Q_Data["営業利益率"],
            text=Q_Data["営業利益率"],
            name="営業利益率",
            yaxis="y2",
            mode="lines+markers+text",
            marker=dict(color="orange", size=50, line_color="peru"),
            opacity=0.8,
            texttemplate="%{y:0.1f}%",
            # marker_color='gold',
            hovertemplate="営業利益率:%{y}",
            textfont=dict(color="white"),
        )
    )
    fig.update_traces(marker_line_width=3)

    fig.update_layout(
        title={
            "text": "Quarterly Revenue&Profit ({})".format(ticker),
            "x": 0.5,
        },
        yaxis=dict(title="Revenue/Profit (M$)", side="left", showgrid=True),
        yaxis2=dict(side="right", overlaying="y", showgrid=False, showticklabels=False),
        xaxis=dict(tickmode="array", tickvals=time_, ticktext=time_, tickangle=-45),
    )

    fig.update_layout(
        title=dict(font=dict(size=25, family="Gravitas")),
        plot_bgcolor=fig_bgColor,
        paper_bgcolor=web_bgColor,
        hoverlabel=dict(font=dict(size=20)),
    )

    fig.update_layout(
        # yaxis_range=[min_-delta*0.5,max_+delta*0.5],
        xaxis_tickangle=-45,
    )
    return fig


@app.callback(
    dash.dependencies.Output("kessan_chart3", "figure"),
    [dash.dependencies.State("stock_select", "value")],
    [dash.dependencies.State("stock_select_text", "value")],
    [Input("scatter_chart", "hoverData")],
)
def update_output(slct_stock, slct_stock_text, hoverData):
    if slct_stock_text != "":
        slct_stock = slct_stock_text.split(",")
    n = hoverData["points"][0]["curveNumber"]
    ticker = slct_stock[n]
    ticker_info = yf.Ticker(ticker)
    y_finance = ticker_info.income_stmt
    Y_DATA = y_finance.loc[["Total Revenue", "Operating Income"], :].transpose()
    time_ = [str(t)[:7] for t in Y_DATA.index]
    Y_DATA["総売上"] = Y_DATA["Total Revenue"].astype(float) / 10**6
    Y_DATA["営業利益"] = Y_DATA["Operating Income"].astype(float) / 10**6
    Y_DATA["営業利益率"] = Y_DATA["営業利益"] / Y_DATA["総売上"] * 100
    max_ = Y_DATA["総売上"].max()
    min_ = Y_DATA["総売上"].min()
    delta = abs(max_ - min_)
    # fig = px.bar(Y_DATA, x='time', y='総売上',title="総売上")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=time_,
            y=Y_DATA["総売上"],
            text=Y_DATA["総売上"],
            name="総売上",
            yaxis="y1",
            # marker_color='gold',
            hovertemplate="総売上:%{y}",
            marker_color="gold",
            marker_line_color="goldenrod",
            marker_line_width=2.5,
            opacity=0.6,
        )
    )

    fig.add_trace(
        go.Bar(
            x=time_,
            y=Y_DATA["営業利益"],
            text=Y_DATA["営業利益"],
            name="営業利益",
            # marker_color='gold',
            hovertemplate="営業利益:%{y}",
            yaxis="y1",
            marker_color="darkseagreen",
            marker_line_color="olivedrab",
            marker_line_width=2.5,
            opacity=0.6,
        )
    )

    fig.add_trace(
        go.Line(
            x=time_,
            y=Y_DATA["営業利益率"],
            text=Y_DATA["営業利益率"],
            name="営業利益率",
            yaxis="y2",
            mode="lines+markers+text",
            marker=dict(color="orange", size=50, line_color="peru"),
            opacity=0.8,
            texttemplate="%{y:0.1f}%",
            # marker_color='gold',
            hovertemplate="営業利益率:%{y}",
            textfont=dict(color="white"),
        )
    )
    fig.update_traces(marker_line_width=3)

    fig.update_layout(
        title={
            "text": "Annual Revenue&Profit ({})".format(ticker),
            "x": 0.5,
        },
        yaxis=dict(title="Revenue/Profit (M$)", side="left", showgrid=True),
        yaxis2=dict(side="right", overlaying="y", showgrid=False, showticklabels=False),
        xaxis=dict(tickmode="array", tickvals=time_, ticktext=time_, tickangle=-45),
    )

    fig.update_layout(
        title=dict(font=dict(size=25, family="Gravitas")),
        plot_bgcolor=fig_bgColor,
        paper_bgcolor=web_bgColor,
        hoverlabel=dict(font=dict(size=20)),
    )

    fig.update_layout(
        # yaxis_range=[min_-delta*0.5,max_+delta*0.5],
        xaxis_tickangle=-45,
    )
    return fig


@app.callback(
    Output("chart_d", "figure"),
    [dash.dependencies.State("stock_select", "value")],
    [dash.dependencies.State("stock_select_text", "value")],
    [dash.dependencies.State("ticker_compared", "value")],
    [dash.dependencies.Input("period_select", "value")],
    [Input("scatter_chart", "hoverData")],
)
def show_img(slct_stock, slct_stock_text, ticker_compared, period_ratio, hoverData):
    if slct_stock_text != "":
        slct_stock = slct_stock_text.split(",")
    n = hoverData["points"][0]["curveNumber"]
    ticker = slct_stock[n]
    data, data_init = calcAnalyticMetrics(ticker, period_ratio)
    if len(ticker_compared) != 0:
        data_compared_list = calcAnalyticMetricsList(ticker_compared, period_ratio)
    else:
        data_compared_list = []

    return createAnalyticMetricsChart(data, data_init, ticker, data_compared_list, ticker_compared)


@app.callback(
    dash.dependencies.Output("output-container-button", "children"),
    [dash.dependencies.Input("button", "n_clicks")],
    [dash.dependencies.State("stock_select", "value")],
    [dash.dependencies.State("stock_select_text", "value")],
)
def update_output(n_clicks, slct_stock, slct_stock_text):
    if slct_stock_text != "":
        slct_stock = slct_stock_text.split(",")
    global value_bf
    if value_bf != slct_stock:
        value_bf = slct_stock
    return "The stocks being analyzed are {}".format(slct_stock)


@app.callback(
    dash.dependencies.Output("ticker_info1", "children"),
    [dash.dependencies.State("stock_select", "value")],
    [dash.dependencies.State("stock_select_text", "value")],
    [Input("scatter_chart", "hoverData")],
)
def update_output(slct_stock, slct_stock_text, hoverData):
    if slct_stock_text != "":
        slct_stock = slct_stock_text.split(",")
    n = hoverData["points"][0]["curveNumber"]
    ticker = slct_stock[n]
    ticker_business = str(stock_table[stock_table["symbol"] == ticker].iloc[0, 1])
    s = "決算情報 : {} ({})".format(ticker, ticker_business)
    return s


@app.callback(
    dash.dependencies.Output("ticker_info2", "children"),
    [dash.dependencies.State("stock_select", "value")],
    [dash.dependencies.State("stock_select_text", "value")],
    [Input("scatter_chart", "hoverData")],
)
def update_output(slct_stock, slct_stock_text, hoverData):
    if slct_stock_text != "":
        slct_stock = slct_stock_text.split(",")
    n = hoverData["points"][0]["curveNumber"]
    ticker = slct_stock[n]
    print(ticker)

    ticker_info = yf.Ticker(ticker)
    calendar = ticker_info.calendar
    ticker_earnDate = calendar["Earnings Date"]

    s = "次回決算日: {}/{}/{}👇👇👇".format(ticker_earnDate[0].year, ticker_earnDate[0].month, ticker_earnDate[0].day)
    return s


if __name__ == "__main__":
    app.run_server(debug=False, use_reloader=False)
