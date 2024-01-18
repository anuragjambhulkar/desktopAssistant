import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.graph_objects as go
from plotly.offline import iplot

#for offline plotting
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 

# %%
tesla = pd.read_csv('C:/Users/AtxGamer/Desktop/datasetsandcodefilesstockmarketprediction/tesla.csv')
tesla.head()

# %%
tesla.info()

# %%
tesla['Date'] = pd.to_datetime(tesla['Date'])

# %%
print(f'Dataframe contains stock prices between {tesla.Date.min()} {tesla.Date.max()}') 
print(f'Total days = {(tesla.Date.max()  - tesla.Date.min()).days} days')

# %%
tesla.describe()

# %%
tesla[['Open','High','Low','Close','Adj Close']].plot(kind='box')

# %%
# Setting the layout for our plot
# Setting the layout for our plotimport plotly.graph_objects as go
from plotly.offline import iplot

# Assuming 'tesla' is your DataFrame with columns 'Date' and 'Close'


# Setting the layout for our plot
layout = go.Layout(
    title='Stock Prices of Tesla',
    xaxis=dict(
        title='Date',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Price',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
tesla_data = go.Scatter(x=tesla['Date'], y=tesla['Close'], mode='lines', name='Tesla Stock')

# Create the plot using go.Figure
plot = go.Figure(data=[tesla_data], layout=layout)

# Use iplot to display the plot in the notebook
iplot(plot)


# %%
#plot(plot) #plotting offline
iplot(plot)

# %%
# Building the regression model
from sklearn.model_selection import train_test_split

#For preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#For model evaluation
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

# %%
#Split the data into train and test sets
X = np.array(tesla.index).reshape(-1,1)
Y = tesla['Close']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

# %%
# Feature scaling
scaler = StandardScaler().fit(X_train)

# %%
from sklearn.linear_model import LinearRegression

# %%
#Creating a linear model
lm = LinearRegression()
lm.fit(X_train, Y_train)

# %%
#Plot actual and predicted values for train dataset
trace0 = go.Scatter(
    x = X_train.T[0],
    y = Y_train,
    mode = 'markers',
    name = 'Actual'
)
trace1 = go.Scatter(
    x = X_train.T[0],
    y = lm.predict(X_train).T,
    mode = 'lines',
    name = 'Predicted'
)
tesla_data = [trace0,trace1]
layout.xaxis.title.text = 'Day'
plot2 = go.Figure(data=tesla_data, layout=layout)

# %%
iplot(plot2)

# %%
#Calculate scores for model evaluation
scores = f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(Y_train, lm.predict(X_train))}\t{r2_score(Y_test, lm.predict(X_test))}
{'MSE'.ljust(10)}{mse(Y_train, lm.predict(X_train))}\t{mse(Y_test, lm.predict(X_test))}
'''
print(scores)

# %%