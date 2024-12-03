# Crypto-Price-and-Twitter-Sentimental-Analysis

We witnessed the history of how social media played a crucial role in the crypto market. For example, in February 2021, Elon Musk's tweet of “Dogecoin is the people's crypto” led to a doubling price of Dogecoin within a week. Inspired by previous research papers and existing AI tools like Kaito, this project aims to explore the potential of using Twitter sentiment analysis, historical prices, and feature analysis to predict Bitcoin prices.

## Collaborator

- Ruby Zhang
- Conghui Duan

## Data Source

Data is sourced using Jenkins task run on a remote server. The server runs data collection script every one hour, and is stored on the remote database. Code for this real-time data pipeline can be found here: [Scheduled-Stock-Price-Scraper](https://github.com/RubyQianru/Scheduled-Stock-Price-Scraper)

- Bitcoin data is sourced from public available real-time crypto price API.
- Twitter data ia sourced from public available Twitter API.
