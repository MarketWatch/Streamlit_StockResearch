
import streamlit as st #importing our required packages and libraries
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
from PIL import Image

image = Image.open('/content/Streamlit banner.png')
st.image(image)
st.markdown("<h1 style='text-align: center; color: White;background-color:#e84343'>Forecasting Stock Prices</h1>", unsafe_allow_html=True)
st.sidebar.header("Forecasting the Stock Prices using Historic Data")
st.sidebar.text("It's a web app that helps user forecast the price of a particular stock using the historic data of the stock.")
st.sidebar.header("Created by - Team MarketWatch")
st.sidebar.text("Member 1 - Aman Sharma")
st.sidebar.text("Member 2 - Bhumil Modi")
st.sidebar.text("Member 3 - Aayush Mishra")


START = "2013-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
stocks = ('GOOG', 'AAPL','MSFT','FB','NFLX','AMZN')
selected_stock = st.selectbox('Select the stock you would like to make predictions for:', stocks)
n_years = st.slider('Years of prediction:', 1, 5)
period = n_years * 365

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data
	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Original data')
st.write(data.head())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecasted data')
st.write(forecast.tail())
    
st.write(f'Forecasted plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)