import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('dashboard_data.csv')

# Color palette (colorblind-safe)
color_map = {
    'Emerging Emitters': '#0077BB',
    'Green Transition Leaders': '#EE7733',
    'Carbon Intensive': '#AA3377'
}

# Initialize app
app = dash.Dash(__name__)
server = app.server

# App layout
app.layout = html.Div([
    dcc.Store(id='selected-country-store', data=None),
    
    # Header
    html.Div([
        html.H1("Carbon Emissions in Europe (2000-2023)", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '5px'}),
        html.P("Interactive Dashboard: K-Means Clustering Analysis of CO2 Emission Patterns",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': '0px'}),
        html.P("Click on any country in any chart to highlight it across all views. Click again to deselect.",
               style={'textAlign': 'center', 'color': '#3498db', 'fontSize': '14px', 'fontWeight': 'bold'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
    
    # Filters Row - Clean single row
    html.Div([
        html.Div([
            html.Label("Select Cluster:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='cluster-filter',
                options=[{'label': 'All Clusters', 'value': 'All'}] + 
                        [{'label': c, 'value': c} for c in df['cluster_label'].unique()],
                value='All',
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'width': '25%', 'display': 'inline-block', 'marginRight': '30px', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Label("Select Year:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': str(y), 'value': y} for y in sorted(df['year'].unique(), reverse=True)],
                value=df['year'].max(),
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '30px', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Span("Selected Country: ", style={'fontWeight': 'bold'}),
            html.Span(id='selected-country-display', children="None", 
                     style={'color': '#e74c3c', 'fontWeight': 'bold'})
        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingTop': '25px'})
        
    ], style={'padding': '15px 20px', 'backgroundColor': '#fff', 'marginBottom': '20px'}),
    
    # Charts Row 1
    html.Div([
        html.Div([
            dcc.Graph(id='map-chart', style={'height': '450px'})
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        html.Div([
            dcc.Graph(id='scatter-chart', style={'height': '500px'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
    ], style={'marginBottom': '20px'}),
    
    # Charts Row 2
    html.Div([
        # Time Series with its own range slider below
        html.Div([
            dcc.Graph(id='time-series-chart', style={'height': '350px'}),
            html.Div([
                html.Label("Year Range:", style={'fontWeight': 'bold', 'fontSize': '12px', 'marginRight': '10px'}),
                dcc.RangeSlider(
                    id='time-series-range-slider',
                    min=df['year'].min(),
                    max=df['year'].max(),
                    value=[df['year'].min(), df['year'].max()],
                    marks={str(y): str(y) for y in range(2000, 2024, 5)},
                    step=1,
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'padding': '0px 40px 10px 40px'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        html.Div([
            dcc.Graph(id='bar-chart', style={'height': '400px'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
    ]),
    
    # Footer
    html.Div([
        html.P("Note: Russia excluded as outlier (1,816 Mt CO2). Clusters based on 6 features: CO2, CO2/capita, CO2/GDP, GDP, Population, and Emission Growth (2000-2023).",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '12px'})
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'marginTop': '20px'})
    
], style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1400px', 'margin': '0 auto', 'padding': '20px'})


@callback(
    Output('selected-country-store', 'data'),
    Output('selected-country-display', 'children'),
    Input('map-chart', 'clickData'),
    Input('scatter-chart', 'clickData'),
    Input('bar-chart', 'clickData'),
    State('selected-country-store', 'data'),
    prevent_initial_call=True
)
def update_selected_country(map_click, scatter_click, bar_click, current_selection):
    
    triggered_id = ctx.triggered_id
    clicked_country = None
    
    if triggered_id == 'map-chart' and map_click:
        clicked_country = map_click['points'][0].get('text') or map_click['points'][0].get('hovertext')
        if clicked_country and len(clicked_country) == 3:
            match = df[df['iso_code'] == clicked_country]['country'].values
            if len(match) > 0:
                clicked_country = match[0]
    
    elif triggered_id == 'scatter-chart' and scatter_click:
        clicked_country = scatter_click['points'][0].get('text') or scatter_click['points'][0].get('hovertext')
    
    elif triggered_id == 'bar-chart' and bar_click:
        clicked_country = bar_click['points'][0].get('y') or bar_click['points'][0].get('label')
    
    if clicked_country == current_selection:
        return None, "None"
    elif clicked_country:
        return clicked_country, clicked_country
    else:
        return current_selection, current_selection if current_selection else "None"


@callback(
    Output('scatter-chart', 'figure'),
    Output('map-chart', 'figure'),
    Output('time-series-chart', 'figure'),
    Output('bar-chart', 'figure'),
    Input('cluster-filter', 'value'),
    Input('year-dropdown', 'value'),
    Input('time-series-range-slider', 'value'),
    Input('selected-country-store', 'data')
)
def update_charts(cluster_filter, selected_year, time_series_range, selected_country):
    
    # Filter for single year (Map, Scatter, Bar)
    df_year = df[df['year'] == selected_year].copy()
    
    # Filter for time series range
    df_time_filtered = df[(df['year'] >= time_series_range[0]) & (df['year'] <= time_series_range[1])]
    
    # Apply cluster filter
    if cluster_filter != 'All':
        df_year = df_year[df_year['cluster_label'] == cluster_filter]
        df_time_filtered = df_time_filtered[df_time_filtered['cluster_label'] == cluster_filter]
    
    # Bubble sizes (normalized within each cluster)
    df_year['bubble_size'] = df_year.groupby('cluster_label')['co2'].transform(
        lambda x: (x / x.max()) * 400 + 40
    )
    
    # =====================
    # Chart 1: Choropleth Map
    # =====================
    if selected_country:
        df_year['opacity'] = df_year['country'].apply(lambda x: 1.0 if x == selected_country else 0.3)
    else:
        df_year['opacity'] = 1.0
    
    map_fig = go.Figure()
    
    for cluster in df_year['cluster_label'].unique():
        cluster_data = df_year[df_year['cluster_label'] == cluster]
        map_fig.add_trace(go.Choropleth(
            locations=cluster_data['iso_code'],
            z=[1] * len(cluster_data),
            text=cluster_data['country'],
            hovertemplate='<b>%{text}</b><br>CO2: %{customdata[0]:.1f} Mt<br>CO2/Capita: %{customdata[1]:.2f} t/person<extra></extra>',
            customdata=np.stack([cluster_data['co2'], cluster_data['co2_per_capita']], axis=-1),
            colorscale=[[0, color_map[cluster]], [1, color_map[cluster]]],
            showscale=False,
            name=cluster,
            marker_line_color=['black' if c == selected_country else 'white' for c in cluster_data['country']],
            marker_line_width=[3 if c == selected_country else 0.5 for c in cluster_data['country']],
            marker_opacity=list(cluster_data['opacity']),
            locationmode='ISO-3',
            showlegend=True
        ))
    
    map_fig.update_layout(
        title=dict(
            text='European Countries by Cluster<br><sup>Clusters based on 2000-2023 emission patterns</sup>',
            font=dict(size=16)
        ),
        geo=dict(
            scope='world',
            showframe=False,
            showcoastlines=True,
            coastlinecolor='gray',
            showland=True,
            landcolor='#f0f0f0',
            showocean=True,
            oceancolor='#d4f1f9',
            showlakes=True,
            lakecolor='#d4f1f9',
            showcountries=True,
            countrycolor='white',
            countrywidth=0.5,
            center=dict(lat=54, lon=15),
            projection_type='mercator',
            lonaxis=dict(range=[-12, 45]),
            lataxis=dict(range=[34, 72])
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        legend_title_text='Cluster',
        uirevision='constant'
    )
    
    # =====================
    # Chart 2: Scatter Plot
    # =====================
    if selected_country:
        df_year['scatter_opacity'] = df_year['country'].apply(lambda x: 1.0 if x == selected_country else 0.3)
        df_year['marker_size'] = df_year.apply(
            lambda x: x['bubble_size'] * 1.3 if x['country'] == selected_country else x['bubble_size'], axis=1
        )
        df_year['line_width'] = df_year['country'].apply(lambda x: 3 if x == selected_country else 1)
        df_year['line_color'] = df_year['country'].apply(lambda x: 'black' if x == selected_country else 'white')
    else:
        df_year['scatter_opacity'] = 0.75
        df_year['marker_size'] = df_year['bubble_size']
        df_year['line_width'] = 1
        df_year['line_color'] = 'white'
    
    scatter_fig = go.Figure()
    
    for cluster in df_year['cluster_label'].unique():
        cluster_data = df_year[df_year['cluster_label'] == cluster]
        max_size_in_cluster = cluster_data['marker_size'].max()
        
        scatter_fig.add_trace(go.Scatter(
            x=cluster_data['co2_per_gdp'],
            y=cluster_data['co2_per_capita'],
            mode='markers',
            name=cluster,
            text=cluster_data['country'],
            hovertemplate='<b>%{text}</b><br>CO2/GDP: %{x:.3f} kg/$<br>CO2/Capita: %{y:.2f} t/person<br>Total CO2: %{customdata:.1f} Mt<extra></extra>',
            customdata=cluster_data['co2'],
            marker=dict(
                size=cluster_data['marker_size'],
                sizemode='area',
                sizeref=2. * max_size_in_cluster / (35. ** 2),
                sizemin=6,
                color=color_map[cluster],
                opacity=list(cluster_data['scatter_opacity']),
                line=dict(
                    width=list(cluster_data['line_width']),
                    color=list(cluster_data['line_color'])
                )
            )
        ))
    
    scatter_fig.update_layout(
        title=dict(
            text=f'Carbon Efficiency Profile ({selected_year})<br><sup>Bubble size represents total CO2 emissions (normalized within each cluster)</sup>',
            font=dict(size=16)
        ),
        xaxis_title='CO2 per GDP (kg/$)',
        yaxis_title='CO2 per Capita (t/person)',
        margin=dict(l=40, r=40, t=60, b=40),
        legend_title_text='Cluster',
        showlegend=True,
        uirevision='constant'
    )
    
    # =====================
    # Chart 3: Time Series
    # =====================
    range_start = time_series_range[0]
    range_end = time_series_range[1]
    
    if selected_country and selected_country in df_time_filtered['country'].values:
        df_country = df_time_filtered[df_time_filtered['country'] == selected_country]
        cluster_of_selected = df_country['cluster_label'].iloc[0]
        df_cluster_avg = df_time_filtered[df_time_filtered['cluster_label'] == cluster_of_selected].groupby('year')['co2'].mean().reset_index()
        
        time_fig = go.Figure()
        
        time_fig.add_trace(go.Scatter(
            x=df_country['year'],
            y=df_country['co2'],
            mode='lines+markers',
            name=selected_country,
            line=dict(width=4, color=color_map[cluster_of_selected]),
            marker=dict(size=8)
        ))
        
        time_fig.add_trace(go.Scatter(
            x=df_cluster_avg['year'],
            y=df_cluster_avg['co2'],
            mode='lines',
            name=f'{cluster_of_selected} (Avg)',
            line=dict(width=2, color=color_map[cluster_of_selected], dash='dash')
        ))
        
        time_fig.update_layout(
            title=f'CO2 Emissions: {selected_country} vs Cluster Average ({range_start}-{range_end})',
            xaxis_title='Year',
            yaxis_title='CO2 Emissions (Mt)',
            margin=dict(l=40, r=40, t=40, b=40),
            legend_title_text='Legend',
            uirevision='constant'
        )
    else:
        df_time = df_time_filtered.groupby(['year', 'cluster_label'])['co2'].sum().reset_index()
        time_fig = px.line(
            df_time,
            x='year',
            y='co2',
            color='cluster_label',
            color_discrete_map=color_map,
            title=f'CO2 Emissions Over Time by Cluster ({range_start}-{range_end})',
            labels={'co2': 'CO2 Emissions (Mt)', 'year': 'Year'}
        )
        time_fig.update_layout(
            margin=dict(l=40, r=40, t=40, b=40),
            legend_title_text='Cluster',
            uirevision='constant'
        )
    
    # =====================
    # Chart 4: Bar Chart
    # =====================
    top_n = min(10, len(df_year))
    df_bar = df_year.nlargest(top_n, 'co2')[['country', 'co2', 'cluster_label']].copy()
    
    if selected_country:
        df_bar['opacity'] = df_bar['country'].apply(lambda x: 1.0 if x == selected_country else 0.4)
        df_bar['line_width'] = df_bar['country'].apply(lambda x: 3 if x == selected_country else 0)
    else:
        df_bar['opacity'] = 1.0
        df_bar['line_width'] = 0
    
    bar_fig = go.Figure()
    
    for cluster in df_bar['cluster_label'].unique():
        cluster_data = df_bar[df_bar['cluster_label'] == cluster]
        bar_fig.add_trace(go.Bar(
            y=cluster_data['country'],
            x=cluster_data['co2'],
            orientation='h',
            name=cluster,
            marker=dict(
                color=color_map[cluster],
                opacity=list(cluster_data['opacity']),
                line=dict(
                    width=list(cluster_data['line_width']),
                    color='black'
                )
            ),
            hovertemplate='<b>%{y}</b><br>CO2: %{x:.1f} Mt<extra></extra>'
        ))
    
    bar_fig.update_layout(
        title=f'Top {top_n} CO2 Emitters ({selected_year})',
        xaxis_title='CO2 Emissions (Mt)',
        yaxis_title='',
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=40, r=40, t=40, b=40),
        legend_title_text='Cluster',
        barmode='stack',
        uirevision='constant'
    )
    
    return scatter_fig, map_fig, time_fig, bar_fig


if __name__ == '__main__':
    app.run(debug=True)