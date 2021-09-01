from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/Пользователь/Downloads/archive/Stocks/aapl.us.txt', parse_dates=['Date'])

df = df[df['Date'] > df['Date'].max() - timedelta(days=365 * 6)]
