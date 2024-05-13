from twikit import Client
import json
import pandas as pd

client = Client('en-US')

with open("login.json", 'r') as file:
  user_info = json.load(file)

client.login(auth_info_1=user_info["username"], password=user_info["password"])
client.save_cookies('cookies.json')
client.load_cookies(path='cookies.json')

# user = client.get_user_by_screen_name("realDonaldTrump")
user = client.get_user_by_screen_name("poland")

tweets = user.get_tweets('Tweets', count=10000)

tweets_to_store = []
for tweet in tweets:
    tweets_to_store.append({
        'created_at': tweet.created_at,
        'favorite_count': tweet.favorite_count,
        'full_text': tweet.full_text,
    })

df = pd.DataFrame(tweets_to_store)
df.to_csv('tweets.csv', index=False)
print(df.sort_values(by='favorite_count', ascending=False))

print(json.dumps(tweets_to_store, indent=4))
