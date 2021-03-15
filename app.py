#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 12:45:50 2021

@author: varunnagrare
"""

from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go
import numpy as np
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import webbrowser


app = Dash(__name__, title='Sentiment Analysis with Insights',
           update_title='Loading...', external_stylesheets=[dbc.themes.BOOTSTRAP])


class NLP_Project:
    def __init__(self, balanced_reviews, scrapped_reviews, app):
        self.balanced_reviews = balanced_reviews
        self.scrapped_reviews = scrapped_reviews
        self.app = app
        self.load_model()

    def load_model(self):
        with open("model_files/trained_model.pkl", 'rb') as file:
            pickle_model = pickle.load(file)
            self.pickle_model = pickle_model
        with open("model_files/vocab.pkl", 'rb') as vocab:
            vocab = pickle.load(vocab)
            self.vocab = vocab

    def check_review(self, reviewText):
        transformer = TfidfTransformer()
        loaded_vec = CountVectorizer(
            decode_error="replace", vocabulary=self.vocab)
        reviewText = transformer.fit_transform(
            loaded_vec.fit_transform([reviewText]))
        return self.pickle_model.predict(reviewText)

    def create_app_ui(self):
        self.balanced_reviews = self.balanced_reviews.dropna()
        # df = df[df['overall'] != 3]
        self.balanced_reviews['Positivity'] = np.where(
            self.balanced_reviews['overall'] > 3, 1, 0)
        labels = ['Positive Reviews', 'Negative Reviews', 'Neutral Reviews']
        values = [self.balanced_reviews[self.balanced_reviews.overall > 3].dropna().shape[0], self.balanced_reviews[self.balanced_reviews.overall < 3].dropna(
        ).shape[0], self.balanced_reviews[self.balanced_reviews.overall == 3].dropna().shape[0]]

        labels1 = ['+ve Reviews', '-ve Reviews']
        values1 = [len(self.balanced_reviews[self.balanced_reviews.Positivity == 1]), len(
            self.balanced_reviews[self.balanced_reviews.Positivity == 0])]

        colors = ['#00cd00', '#d80000', '#a6a6a6']

        main_layout = dbc.Container(
            dbc.Jumbotron(
                [
                    html.H1(id='heading1', children='Sentiment Analysis with Insights',
                            className='display-3 mb-4', style={'font': 'sans-seriff', 'font-weight': 'bold', 'font-size': '50px', 'color': 'black'}),
                    
                    html.P(id='heading5', children='Distribution of reviews based on filtered data',
                           className='display-3 mb-4', style={'font': 'sans-seriff', 'font-weight': 'bold', 'font-size': '30px', 'color': 'black'}),
                    
                    dbc.Container(
                        dcc.Loading(
                            dcc.Graph(
                                figure={'data': [go.Pie(labels=labels, values=values, hole=.3, pull=[0.2, 0, 0], textinfo='value', marker=dict(colors=colors, line=dict(color='#000000', width=2)))],
                                        'layout': go.Layout(height=600, width=1000, autosize=False)
                                        }
                            )
                        ),
                        className='d-flex justify-content-center'
                    ),

                    html.Hr(),
                    
                    html.P(id='heading4', children='The Positivity Measure',
                           className='display-3 mb-4', style={'font': 'sans-seriff', 'font-weight': 'bold', 'font-size': '30px', 'color': 'black'}),
                    
                    dbc.Container(
                        dcc.Loading(
                            dcc.Graph(
                                id='example-graph',
                                figure={
                                    'data': [
                                        go.Bar(y=labels1, x=values1, orientation='h', marker=dict(
                                            color="MediumPurple"))
                                    ],
                                    'layout': go.Layout(xaxis={'title': 'Sentiments'}, yaxis={'title': 'Emotions'}),
                                }
                            )
                        ),
                    ),

                    html.Hr(),

                    html.P(id='heading2', children='Feel as you type!',
                           className='display-3 mb-4', style={'font': 'sans-seriff', 'font-weight': 'bold', 'font-size': '30px', 'color': 'black'}),
                    dbc.Textarea(id='textarea', className="mb-3", placeholder="Enter a review",
                                 value='', style={'resize': 'none'}),

                    html.Div(id='result'),
                    html.Hr(),

                    html.P(id='heading3', children='Scrapped Etsy Review Sentiments',
                           className='display-3 mb-4', style={'font': 'sans-serif', 'font-weight': 'bold', 'font-size': '30px', 'color': 'black'}),

                    dbc.Container([
                        dcc.Dropdown(
                            id='dropdown',
                            placeholder='See what people think',
                            options=[{'label': i[:100] + "...", 'value': i}
                                     for i in self.scrapped_reviews.reviews],
                            value=self.balanced_reviews.reviewText[0],
                            style={'margin':'10px'}

                        )
                    ]),

                    html.Div(id='result1')
                ],
                className='text-center'
            ),
            className='mt-4'
        )
        return main_layout


@app.callback(
    Output('result', 'children'),
    [
        Input('textarea', 'value')
    ]
)
def update_app_ui(value):
    rev = NLP_Project(None, None, app)
    rev.load_model()
    result_list = rev.check_review(value)[0]

    if (result_list == 0):
        return dbc.Alert("Ahh too much Negativity!", color="danger")
    elif (result_list == 1):
        return dbc.Alert("That's GREAT!", color="success")
    elif (result_list == ''):
        return dbc.Alert("No review yet", color="dark")


@app.callback(
    Output('result1', 'children'),
    [
        Input('dropdown', 'value')
    ]
)
def update_dropdown(value):
    rev = NLP_Project(None, None, app)
    rev.load_model()
    result_list = rev.check_review(value)[0]

    if (result_list == 0):
        return dbc.Alert("Ahh too much Negativity!", color="danger")
    elif (result_list == 1):
        return dbc.Alert("That's GREAT!", color="success")
    elif (result_list == None):
        return dbc.Alert("No review yet", color="dark")


def main():
    balanced_reviews = pd.read_csv("balanced_reviews.csv")
    scrapped_reviews = pd.read_csv("etsy_swimwear_reviews.csv")
    app.layout = NLP_Project(
        balanced_reviews, scrapped_reviews, app).create_app_ui()
    app.run_server(debug=True)


if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:8050/")
    main()
