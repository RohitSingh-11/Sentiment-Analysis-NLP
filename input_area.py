import tweepy_streamer as ts
h_tag = input("enter @ tag ")

twitter_client = TwitterClient()
tweet_analyzer = TweetAnalyzer()

api = twitter_client.get_twitter_client_api()

tweets = api.user_timeline(screen_name="h_tag",count=10)

df = tweet_analyzer.tweets_to_data_frame(tweets)
df['sentiment'] = np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in df['tweets']])

print(df.head(10))
time_likes = pd.Series(data=df['likes'].values,index=df['date'])
time_likes.plot(figsize=(4,3),color='r')
plt.ylabel("likes")
plt.xlabel("date")
plt.show()
sen = df["sentiment"]
sen = np.asarray(sen)
print(sen)
