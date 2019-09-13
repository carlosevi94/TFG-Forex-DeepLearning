import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from plotly import subplots

import pandas as pd
import datetime


import mysql.connector as sql

BD = 'forex_values'

USER = 'root'

PASSWORD = 'root'

CURRENCIES = ['EUR/USD', 'BTC/USD']

external_css = ["https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
                "https://fonts.googleapis.com/css?family=Raleway:400,400i,700,700i",
                "https://fonts.googleapis.com/css?family=Product+Sans:400,400i,700,700i"]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(
    'forex-live-dash',
    external_stylesheets=external_stylesheets
)
server = app.server


# Devuelve el ultimo valor de predicciones
def get_realtime(currency_pair):

    currency = currency_pair.replace('/', '_').lower()
    db_connection = sql.connect(host='localhost', database=BD, user=USER, password=PASSWORD, port=9306)
    res = pd.read_sql_query('SELECT id, value from realtime_%s order by id desc LIMIT 1' % currency,
                            db_connection, coerce_float=False)

    res = res.reset_index()

    res = res.drop(['id'], axis=1)

    res = res.loc[0]

    db_connection.close()

    return res, currency_pair  # returns dataset row and index of row


# Crea el html donde se ven los valores
def get_row(data):
    current_row = data[0]
    index = current_row[0]
    currency = data[1]

    return html.Div(
        children=[
            # Summary
            html.Div(
                id=currency + "summary",
                className="row ",
                n_clicks=0,
                children=[
                    html.Div(
                        id=currency + "row",
                        className="row",
                        children=[
                            html.P(
                                currency,  # currency pair name
                                id=currency,
                                className="three-col",
                            ),
                            html.P(
                                current_row[1].round(7),  # Bid value
                                id=currency + "value",
                                className="three-col", style={'width': '5%'},
                            ),
                            html.P(
                                datetime.datetime.now().strftime("%H:%M"),  # Bid date
                                id=currency + "date",
                                className="three-col", style={'width': '5%'},
                            ),
                            html.Div(
                                index,
                                id=currency
                                + "index",  # we save index of row in hidden div
                                style={"display": "none"},
                            ),
                        ],  style={'display': 'block', 'margin': 'auto', 'text-align': 'center'}
                    )
                ],
            ),
        ]
    )


def get_color(a, b, date, color):
    if not color.get('color'):
        if a == b:
            return "white"
        elif a > b:
            return "#45df7e"
        else:
            return "#da5657"
    else:
        if datetime.datetime.now().strftime("%H:%M") == date:
            return color.get('color')
        else:
            if a == b:
                return "white"
            elif a > b:
                return "#45df7e"
            else:
                return "#da5657"


def replace_row(currency_pair, index, value, date, color):
    index = index + 1  # index of new data row
    new_row = get_realtime(currency_pair)[0]

    return [
        html.P(
            currency_pair, id=currency_pair, className="three-col", style={'width': '10%'} # currency pair name
        ),
        html.P(
            new_row[1].round(7),  # Bid value
            id=currency_pair + 'value',
            className="three-col",
            style={"color": get_color(round(new_row[1], 7), value, date, color), 'width': '5%'},
        ),
        html.P(
            datetime.datetime.now().strftime("%H:%M"),  # Bid value
            id=currency_pair + 'date',
            className="three-col", style={'width': '5%'},
        ),
        html.Div(
            index, id=currency_pair + "index", style={"display": "none"}
        ),  # save index in hidden div
    ]


app.layout = html.Div([
    html.Div([
        html.Img(src=app.get_asset_url('logo.png'), style={'width': '30%', 'display': 'block', 'margin': 'auto'})
    ], className='banner'),
    html.Div([
        html.Div(
            className="div-currency-toggles",
            children=[
                # html.P(
                #     className="six-col",
                #     children='',
                # ),
                # html.P(className="three-col", children="Value"),
                html.Div(
                    id="pairs",
                    className="div-bid-ask",
                    children=[
                        get_row(get_realtime(pair))
                        for pair in CURRENCIES
                    ],
                ),
            ],
        ),
        html.Div([
            # html.Label("Currency",
            #            style={'padding-left': '2px'}),
            dcc.Dropdown(
                id='currency',
                options=[{'label': 'EUR/USD', 'value': 'eur_usd'}, {'label': 'BTC/USD', 'value': 'btc_usd'}],
                value='eur_usd',
                placeholder="Currency",
                style={'width': '100%', 'display': 'inline-block'}
            ),
        ], style={'width': '100%', 'display': 'inline-block'}),
        html.Br(),
        html.Br(),
        dcc.Tabs(id="tabs", children=[
            dcc.Tab(label='Recurrent Neural Network ', value='rnn', selected_style={'backgroundColor': '#21252C', 'color': 'white'}),
            dcc.Tab(label='Convolutional neural network', value='cnn', selected_style={'backgroundColor': '#21252C', 'color': 'white'}),
        ], value="rnn", colors={'border': 'gray', 'background': '#21252C'}),
        html.Div([
            dcc.Graph(id='graph'),
        ], className='twelve columns eur-usd-graph'),
        html.Div([
            dcc.Dropdown(id='type', options=[
                {"label": "Line", "value": "line_trace"},
                {"label": "Mountain", "value": "area_trace"},
                {
                    "label": "Colored bar",
                    "value": "colored_bar_trace",
                },
            ], value="line_trace", clearable=False, style={'width': '98%'}),
        ], className='twelve columns', style={'display': 'inline-block', 'width': '40%'}),
        html.Div([
            dcc.Dropdown(id='metric', options=[
                {
                    "label": "Accumulation/D",
                    "value": "accumulation_trace",
                },
                {
                    "label": "Bollinger bands",
                    "value": "bollinger_trace",
                },
                {"label": "MA", "value": "moving_average_trace"},
                {"label": "EMA", "value": "e_moving_average_trace"},
                {"label": "CCI", "value": "cci_trace"},
                {"label": "ROC", "value": "roc_trace"},
                {"label": "Pivot points", "value": "pp_trace"},
                {
                    "label": "Stochastic oscillator",
                    "value": "stoc_trace",
                },
                {
                    "label": "Momentum indicator",
                    "value": "mom_trace",
                },
            ], clearable=True, style={'width': '100%'}, multi=True),
        ], className='twelve columns', style={'display': 'inline-block', 'width': '60%'}),
        html.Div([
            dcc.Graph(id='train'),
        ], className='twelve columns'),
        dcc.Interval(id='update', interval=10 * 1000, n_intervals=0),
    ], className='row eur-usd-graph-row'),

], style={'padding': '0px 10px 15px 10px',
          'marginLeft': '2%', 'marginRight': '2%',
          'boxShadow': '0px 0px 5px 5px rgba(204,204,204,0.4)'})


@app.callback(Output('graph', 'figure'), [Input('update', 'n_intervals'), Input('currency', 'value'), Input('tabs', 'value')])
def plot_graph(interval, currency, tab):
    # Open DB connection
    db_connection = sql.connect(host='localhost', database=BD, user=USER, password=PASSWORD, port=9306)

    data = list()

    if currency:
        # Historical
        df = pd.read_sql_query('SELECT id, value from historic_%s order by id desc LIMIT 15' % currency, db_connection)

        df = df.sort_values('id')

        historic = go.Scatter(
            y=df['value'],
            x=df['id'],
            name="Historical"
        )
        data.append(historic)

        # Realtime
        df = pd.read_sql_query('SELECT id, value from realtime_%s order by id desc LIMIT 15' % currency, db_connection)

        df = df.sort_values('id')

        realtime = go.Scatter(
            y=df['value'],
            x=df['id'],
            name="Realtime"
        )
        data.append(realtime)

        # NET
        df = pd.read_sql_query('SELECT id, value from prediction_%s_%s order by id desc LIMIT 46' % (currency, tab),
                               db_connection)

        df = df.sort_values('id')

        prediction = go.Scatter(
            y=df['value'],
            x=df['id'],
            name="Prediction"
        )
        data.append(prediction)

        db_connection.close()

        res = go.Figure(data=data)

        res["layout"][
            "uirevision"
        ] = "The User is always right"  # Ensures zoom on graph is the same on update
        res["layout"]["margin"] = {"t": 50, "l": 50, "b": 50, "r": 25}
        res["layout"]["autosize"] = True
        res["layout"]["height"] = 400
        res["layout"]["xaxis"]["rangeslider"]["visible"] = False
        # res["layout"]["xaxis"]["tickformat"] = "%H:%M"
        res["layout"]["yaxis"]["showgrid"] = True
        res["layout"]["yaxis"]["gridcolor"] = "#3E3F40"
        res["layout"]["yaxis"]["gridwidth"] = 1
        res["layout"].update(paper_bgcolor="#21252C", plot_bgcolor="#21252C")

    # Close DB connection

    return res


@app.callback(Output('train', 'figure'), [Input('currency', 'value'), Input('type', 'value'), Input('metric', 'value')])
def plot_train(currency, type_plot, metric):
    # Open DB connection
    db_connection = sql.connect(host='localhost', database=BD, user=USER, password=PASSWORD, port=9306)

    # Train
    df = pd.read_sql_query('SELECT id, value from historic_%s order by id desc limit 100000' % currency, db_connection)

    df = df.sort_values('id')
    df = df.set_index('id')
    df_ohlc = df.resample('5Min').ohlc()
    if type_plot == 'colored_bar_trace':
        # print(df.head())
        # print(df.columns)
        historic = go.Ohlc(
            x=df_ohlc.index,
            open=df_ohlc['value']["open"],
            high=df_ohlc['value']["high"],
            low=df_ohlc['value']["low"],
            close=df_ohlc['value']["close"],
            name='Historical'
        )
    else:
        if type_plot == 'area_trace':
            fill = 'toself'
        else:
            fill = 'none'

        historic = go.Scatter(
            y=df['value'],
            x=df.index,
            name="Historical",
            fill=fill
        )

    db_connection.close()

    subplot_traces = [  # first row traces
        "accumulation_trace",
        "cci_trace",
        "roc_trace",
        "stoc_trace",
        "mom_trace",
    ]

    selected_subplots_studies = []
    selected_first_row_studies = []
    row = 1


    # Dibuja metricas
    if metric:
        for study in metric:
            if study in subplot_traces:
                row += 1  # increment number of rows only if the study needs a subplot
                selected_subplots_studies.append(study)
            else:
                selected_first_row_studies.append(study)

    res = subplots.make_subplots(
        rows=row,
        shared_xaxes=True,
        shared_yaxes=True,
        cols=1,
        print_grid=True,
        # vertical_spacing=0.12,
    )

    res.append_trace(historic, 1, 1)

    for study in selected_first_row_studies:
        res = eval(study)(df_ohlc, res)

    row = 1
    # Plot trace on new row
    for study in selected_subplots_studies:
        row += 1
        res.append_trace(eval(study)(df_ohlc), row, 1)

    res["layout"][
        "uirevision"
    ] = "The User is always right"  # Ensures zoom on graph is the same on update
    res["layout"]["margin"] = {"t": 50, "l": 50, "b": 50, "r": 25}
    res["layout"]["autosize"] = True
    res["layout"]["height"] = 400
    res["layout"]["xaxis"]["rangeslider"]["visible"] = False
    # res["layout"]["xaxis"]["tickformat"] = "%H:%M"
    res["layout"]["yaxis"]["showgrid"] = True
    res["layout"]["yaxis"]["gridcolor"] = "#3E3F40"
    res["layout"]["yaxis"]["gridwidth"] = 1
    res["layout"].update(paper_bgcolor="#21252C", plot_bgcolor="#21252C")
    return res


def generate_ask_bid_row_callback(pair):
    def output_callback(n, i, value, date, color):
        return replace_row(pair, int(i), float(value), date, color)

    return output_callback


# METRICAS

# Moving average
def moving_average_trace(df, fig):
    df2 = df.rolling(window=5).mean()
    trace = go.Scatter(
        x=df2.index, y=df2["value"]["close"], mode="lines", showlegend=False, name="MA"
    )
    fig.append_trace(trace, 1, 1)  # plot in first row
    return fig


# Exponential moving average
def e_moving_average_trace(df, fig):
    df2 = df.rolling(window=20).mean()
    trace = go.Scatter(
        x=df2.index, y=df2['value']["close"], mode="lines", showlegend=False, name="EMA"
    )
    fig.append_trace(trace, 1, 1)  # plot in first row
    return fig


# Bollinger Bands
def bollinger_trace(df, fig, window_size=10, num_of_std=5):
    price = df['value']["close"]
    rolling_mean = price.rolling(window=window_size).mean()
    rolling_std = price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)

    trace = go.Scatter(
        x=df.index, y=upper_band, mode="lines", showlegend=False, name="BB_upper"
    )

    trace2 = go.Scatter(
        x=df.index, y=rolling_mean, mode="lines", showlegend=False, name="BB_mean"
    )

    trace3 = go.Scatter(
        x=df.index, y=lower_band, mode="lines", showlegend=False, name="BB_lower"
    )

    fig.append_trace(trace, 1, 1)  # plot in first row
    fig.append_trace(trace2, 1, 1)  # plot in first row
    fig.append_trace(trace3, 1, 1)  # plot in first row
    return fig


# Accumulation Distribution
def accumulation_trace(df):
    df["volume"] = ((df['value']["close"] - df['value']["low"]) - (df['value']["high"] - df['value']["close"])) / (
        df['value']["high"] - df['value']["low"]
    )
    trace = go.Scatter(
        x=df.index, y=df["volume"], mode="lines", showlegend=False, name="Accumulation"
    )
    return trace


# Commodity Channel Index
def cci_trace(df, ndays=5):
    TP = (df['value']["high"] + df['value']["low"] + df['value']["close"]) / 3
    CCI = pd.Series(
        (TP - TP.rolling(window=10, center=False).mean())
        / (0.015 * TP.rolling(window=10, center=False).std()),
        name="cci",
    )
    trace = go.Scatter(x=df.index, y=CCI, mode="lines", showlegend=False, name="CCI")
    return trace


# Price Rate of Change
def roc_trace(df, ndays=5):
    N = df['value']["close"].diff(ndays)
    D = df['value']["close"].shift(ndays)
    ROC = pd.Series(N / D, name="roc")
    trace = go.Scatter(x=df.index, y=ROC, mode="lines", showlegend=False, name="ROC")
    return trace


# Stochastic oscillator %K
def stoc_trace(df):
    SOk = pd.Series((df['value']["close"] - df['value']["low"]) / (df['value']["high"] - df['value']["low"]), name="SO%k")
    trace = go.Scatter(x=df.index, y=SOk, mode="lines", showlegend=False, name="SO%k")
    return trace


# Momentum
def mom_trace(df, n=5):
    M = pd.Series(df['value']["close"].diff(n), name="Momentum_" + str(n))
    trace = go.Scatter(x=df.index, y=M, mode="lines", showlegend=False, name="MOM")
    return trace


# Pivot points
def pp_trace(df, fig):
    PP = pd.Series((df['value']["high"] + df['value']["low"] + df['value']["close"]) / 3)
    R1 = pd.Series(2 * PP - df['value']["low"])
    S1 = pd.Series(2 * PP - df['value']["high"])
    R2 = pd.Series(PP + df['value']["high"] - df['value']["low"])
    S2 = pd.Series(PP - df['value']["high"] + df['value']["low"])
    R3 = pd.Series(df['value']["high"] + 2 * (PP - df['value']["low"]))
    S3 = pd.Series(df['value']["low"] - 2 * (df['value']["high"] - PP))
    trace = go.Scatter(x=df.index, y=PP, mode="lines", showlegend=False, name="PP")
    trace1 = go.Scatter(x=df.index, y=R1, mode="lines", showlegend=False, name="R1")
    trace2 = go.Scatter(x=df.index, y=S1, mode="lines", showlegend=False, name="S1")
    trace3 = go.Scatter(x=df.index, y=R2, mode="lines", showlegend=False, name="R2")
    trace4 = go.Scatter(x=df.index, y=S2, mode="lines", showlegend=False, name="S2")
    trace5 = go.Scatter(x=df.index, y=R3, mode="lines", showlegend=False, name="R3")
    trace6 = go.Scatter(x=df.index, y=S3, mode="lines", showlegend=False, name="S3")
    fig.append_trace(trace, 1, 1)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 1)
    fig.append_trace(trace3, 1, 1)
    fig.append_trace(trace4, 1, 1)
    fig.append_trace(trace5, 1, 1)
    fig.append_trace(trace6, 1, 1)
    return fig


for pair in CURRENCIES:
    app.callback(
        Output(pair + "row", "children"),
        [Input("update", "n_intervals")],
        [
            State(pair + "index", "children"),
            State(pair + "value", "children"),
            State(pair + "date", "children"),
            State(pair + "value", "style"),
        ],
    )(generate_ask_bid_row_callback(pair))


if __name__ == '__main__':
    app.run_server(debug=False)
