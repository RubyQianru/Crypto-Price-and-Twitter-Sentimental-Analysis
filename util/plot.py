import matplotlib.pyplot as plt

def sentiment_price_scatter(
  price, sentiment_1, sentiment_2,
  title_1:str, x_label_1:str, y_label_1:str,
  title_2:str, x_label_2:str, y_label_2:str,
  ):

  f, ax = plt.subplots(1, 2, figsize=(15, 6))
  ax[0].scatter(price, sentiment_1)
  ax[0].set_xlabel(x_label_1)
  ax[0].set_ylabel(y_label_1)
  ax[0].set_title(title_1)

  ax[1].scatter(price, sentiment_2)
  ax[1].set_xlabel(x_label_2)
  ax[1].set_ylabel(y_label_2)
  ax[1].set_title(title_2)

  plt.gcf().autofmt_xdate()
  plt.tight_layout()
  plt.legend()
  plt.show()