import pandas as pd
import numpy as np

import plotly.graph_objs as go
from plotly.offline import plot,iplot

aapl = pd.read_csv('C:/Users/admin/PycharmProjects/web scraping/historical_data.csv')
'''
#aapl.head()
aapl.info() #non null values
aapl['Date']=pd.to_datetime(aapl['Date'])
print(f'dataframe between {aapl.Date.min()} and {aapl.Date.max()}')
print(f'total days = {(aapl.Date.max()-aapl.Date.min().days)} days')
'''
aapl.describe()
layout =go.Layout(
    title='stock price of aaple',
    xaxis=dict(
        title='date',
        titlefont=dict(
            family='Courier New,monospace',
            size=18,
            color='#7f7f7f'

        )
    ),
      yaxis=dict(
            title='Price',
            titlefont=dict(
                family='Courier New,monospace',
                size=18,
                color='#7f7f7f'

            )
        )

)
aapl_data=[{'x':aapl['Date'],'y':aapl['Close']}]
plot_figure = go.Figure(data=aapl_data, layout=layout)
plot(plot_figure, filename='aapl_stock_price.html')

