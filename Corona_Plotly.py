import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
import re

# Download NLTK stopwords
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('Corona_NLP_train.csv', encoding='latin')


# Function to remove URLs from text
def remove_urls(text):
    return re.sub(r'http\S+|www\S+', '', text)


# Preprocess the data
stop_words = set(stopwords.words('english'))


# Function to remove URLs from text
def remove_urls(text):
    return re.sub(r'http\S+|www\S+', '', text)


# Preprocess the data
df['cleaned_tweet'] = df['OriginalTweet'].apply(lambda x: remove_urls(x))  # Call remove_urls function first
df['cleaned_tweet'] = df['cleaned_tweet'].apply(
    lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

# Create the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1('COVID-19 Tweet Analysis'),
    dcc.Tabs([
        dcc.Tab(label='Sentiment Analysis', children=[
            dcc.Graph(id='sentiment-pie-chart'),
            dcc.Graph(id='sentiment-bar-chart'),
            dcc.Graph(id='tweet-frequency-line-chart'),
            dcc.Graph(id='location-bar-chart')
        ]),
        dcc.Tab(label='Word Cloud', children=[
            html.Img(id='word-cloud-image')
        ])
    ])
])


# Define the callback functions
@app.callback(
    Output('sentiment-pie-chart', 'figure'),
    [Input('sentiment-pie-chart', 'id')])
def update_sentiment_pie_chart(id):
    # Utilize the function to generate sentiment pie chart
    fig = generate_sentiment_pie_chart()
    return fig


@app.callback(
    Output('sentiment-bar-chart', 'figure'),
    [Input('sentiment-bar-chart', 'id')])
def update_sentiment_bar_chart(id):
    # Utilize the function to generate sentiment bar chart
    fig = generate_sentiment_bar_chart()
    return fig


@app.callback(
    Output('tweet-frequency-line-chart', 'figure'),
    [Input('tweet-frequency-line-chart', 'id')])
def update_tweet_frequency_line_chart(id):
    # Utilize the function to generate tweet frequency line chart
    fig = generate_tweet_frequency_line_chart()
    return fig


@app.callback(
    Output('location-bar-chart', 'figure'),
    [Input('location-bar-chart', 'id')])
def update_location_bar_chart(id):
    # Utilize the function to generate location bar chart
    fig = generate_location_bar_chart()
    return fig


@app.callback(
    Output('word-cloud-image', 'src'),
    [Input('word-cloud-image', 'id')])
def update_word_cloud(id):
    text = ' '.join(df['cleaned_tweet'])
    wordcloud = WordCloud().generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('word_cloud.png', bbox_inches='tight')
    with open('word_cloud.png', 'rb') as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode()
    return 'data:image/png;base64,{}'.format(encoded_image)


# Define the sentiment pie chart function
def generate_sentiment_pie_chart():
    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    color_mapping = {
        'Positive': 'lightgreen',
        'Extremely Positive': 'darkgreen',
        'Negative': 'orange',
        'Extremely Negative': 'darkred',
        'Neutral': 'lightgrey'
    }
    sentiment_counts['Color'] = sentiment_counts['Sentiment'].map(color_mapping)
    fig_pie = px.pie(sentiment_counts,
                     values='Count',
                     names='Sentiment',
                     color='Sentiment',
                     color_discrete_map=color_mapping,
                     title='Sentiment Distribution')
    return fig_pie


# Define the sentiment bar chart function
def generate_sentiment_bar_chart():
    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig_bar = px.bar(sentiment_counts, x='Sentiment', y='Count',
                     labels={'Sentiment': 'Sentiment', 'Count': 'Tweet Count'},
                     color='Sentiment',
                     color_discrete_map={
                         'Positive': 'lightgreen',
                         'Extremely Positive': 'darkgreen',
                         'Negative': 'orange',
                         'Extremely Negative': 'darkred',
                         'Neutral': 'lightgrey'
                     },
                     title='Sentiment Distribution')
    return fig_bar


# Define the tweet frequency line chart function
def generate_tweet_frequency_line_chart():
    tweet_freq_over_time = df.groupby(df['TweetAt'].dt.date).size().reset_index(name='Count')
    line_chart_tweet_freq = px.line(tweet_freq_over_time, x='TweetAt', y='Count', title='Tweet Frequency Over Time',
                                    labels={'TweetAt': 'Date', 'Count': 'Tweet Count'})
    return line_chart_tweet_freq


# Define the location bar chart function
def generate_location_bar_chart():
    location_counts = df['Location'].value_counts().nlargest(10)
    bar_chart_location = px.bar(x=location_counts.values, y=location_counts.index,
                                orientation='h', title='Top 10 Tweet Locations',
                                labels={'x': 'Count', 'y': 'Location'})
    return bar_chart_location


if __name__ == '__main__':
    app.run_server(debug=True)
