# Crypto-Price-and-Twitter-Sentimental-Analysis

Predicting stock market movements is a well-known problem of interest. Now-a-days social media is perfectly representing the public sentiment and opinion about current events. Especially, twitter has attracted a lot of attention from researchers for studying the public sentiments. We have seen that stock market prediction on the basis of public sentiments expressed on twitter has been an intriguing field of research.

We report on the use of sentiment analysis on news and social media to analyze and predict the price of Bitcoin. Bitcoin is the leading cryptocurrency and has the highest market capitalization among digital currencies. Predicting Bitcoin values may help understand and predict potential market movement and future growth of the technology. Unlike (mostly) repeating phenomena like weather, cryptocurrency values do not follow a repeating pattern and mere past value of Bitcoin does not reveal any secret of future Bitcoin value. Humans follow general sentiments and technical analysis to invest in the market. Hence considering people’s sentiment can give a good degree of prediction. We focus on using social sentiment as a feature to predict future Bitcoin value.

We witnessed the history of how social media played a crucial role in the crypto market. For example, in February 2021, Elon Musk's tweet of “Dogecoin is the people's crypto” led to a doubling price of Dogecoin within a week. Inspired by previous research papers and existing AI tools like Kaito, this project aims to explore the potential of using Twitter sentiment analysis, historical prices, and feature analysis to predict Bitcoin prices.

The project is consituted of 5 parts:

- Bitcoin Price Analysis
- Twitter Sentiment Analysis
- Correlation between Bitcoin price and Twitter Sentiment
- Random Forest Prediction
- Conclusion and Next Steps

## Collaborator

- Ruby Zhang
- Conghui Duan

## Data Source

Data is sourced using Jenkins task run on a remote server. The server runs data collection script every one hour, and is stored on the remote database. Code for this real-time data pipeline can be found here: [Scheduled-Stock-Price-Scraper](https://github.com/RubyQianru/Scheduled-Stock-Price-Scraper)

- Bitcoin data is sourced from public available real-time crypto price API.
- Twitter data ia sourced from public available Twitter API.

### Conclusions

In this project, we load and process the dataset of Bitcoin price and Twitter Sentiment and try to find what features contribute to the target Bitcoin prices. Based on the ML training result, we come to the conclusion that historical price, sentiment and technical indicators all contribute partly to the movement of target price.

- Historical Prices: Historical price data plays the most significant role in predicting Bitcoin's target price, as it directly reflects past market trends.
- Sentiment: Public sentiment derived from Twitter (e.g., positivity, neutrality, and confidence metrics) contributes meaningfully to price movements, highlighting the importance of capturing the market's psychological state.
- Technical Indicators: Metrics such as moving averages and volatility also impact price prediction by capturing trends and market momentum.

While historical price and technical indicators are straightforward to compute, sentiment analysis offers unique opportunities to uncover potential alpha by analyzing public opinions from diverse sources.

### Next Steps

- Expand the dataset: Incorporate a longer time period with more data to improve model robustness and reduce overfitting.
- Explore Advanced Models: Test LSTM and ARIMA models, which are designed for time-series data and can account for trend-specific patterns and lagged relationships.
- Focus on Predicting Price Direction: Instead of predicting exact prices, shift focus to predicting price movement direction (up or down), which can serve as actionable trading advice.
- Diversify Sentiment Sources: Extend sentiment analysis to other platforms such as Google News, Reddit, and financial blogs for a broader understanding of market sentiment.
