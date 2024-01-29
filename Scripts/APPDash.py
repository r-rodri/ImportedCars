# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:00:52 2023

@author: Rodrigo
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.io as pio
import os

wd = os.getcwd()
wd = os.path.abspath(os.path.join(wd, os.pardir))
path = os.chdir(wd)
path1 = os.path.join(wd,'Updated Data','Merged.csv')

data = pd.read_csv(path1)

# Set the default renderer to 'browser'
pio.renderers.default = 'browser'

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='modelo-dropdown',
        options=[{'label': modelo, 'value': modelo} for modelo in data['Modelo'].unique()],
        value='420',  # Default selected value
        clearable=False
    ),
    dcc.Graph(id='box-plot')
])

@app.callback(
    Output('box-plot', 'figure'),
    [Input('modelo-dropdown', 'value')]
)
def update_graph(selected_modelo):
    # Filter data based on selected 'modelo'
    filtered_data = data[data['Modelo'] == selected_modelo]

    # Create a box plot using Plotly Express for the selected 'modelo'
    fig = px.box(filtered_data, y="Preco", color='Portugal', points='all', title=f'Box Plot for {selected_modelo}')
    fig.update_xaxes(showticklabels=False)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
