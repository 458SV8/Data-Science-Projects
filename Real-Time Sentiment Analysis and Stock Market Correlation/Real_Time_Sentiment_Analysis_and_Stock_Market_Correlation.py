import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px

tweets_path = r"C:\Users\Shrey\OneDrive\Desktop\data science projects\Real-Time Sentiment Analysis and Stock Market Correlation\data set\archive\stock_tweets.csv"
stock_data_path = r"C:\Users\Shrey\OneDrive\Desktop\data science projects\Real-Time Sentiment Analysis and Stock Market Correlation\data set\archive\stock_yfinance_data.csv"

tweets = pd.read_csv(tweets_path)
stock_data = pd.read_csv(stock_data_path)

def clean_tweet(tweet):
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = tweet.strip()
    return tweet

tweets['cleaned_text'] = tweets['Tweet'].apply(clean_tweet)

tweets['Date'] = pd.to_datetime(tweets['Date']).dt.tz_localize(None)
stock_data['Date'] = pd.to_datetime(stock_data['Date'])


stock_name = 'TSLA'
stock_data = stock_data[stock_data['Stock Name'] == stock_name]
tweets = tweets[tweets['Stock Name'] == stock_name]


analyzer = SentimentIntensityAnalyzer()
tweets['Sentiment'] = tweets['cleaned_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])


merged_data = pd.merge_asof(stock_data.sort_values('Date'),
                            tweets.sort_values('Date'),
                            left_on='Date', right_on='Date',
                            direction='nearest')


correlation = merged_data['Sentiment'].corr(merged_data['Close'])

print(f"\nCorrelation between sentiment and stock price: {correlation:.2f}")


fig = px.scatter(merged_data, x='Sentiment', y='Close', 
                 title=f'Sentiment vs Stock Price (Correlation: {correlation:.2f})')
fig.show()
