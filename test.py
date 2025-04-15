import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

def fetch_and_plot_dxy_oil():
    # Fetch 3 months of data (daily)
    dxy = yf.Ticker("DX-Y.NYB")
    oil = yf.Ticker("CL=F")
    
    dxy_hist = dxy.history(period="3mo", interval="1d")
    oil_hist = oil.history(period="3mo", interval="1d")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(dxy_hist.index, dxy_hist['Close'], label='US Dollar Index (DXY)', color='blue')
    plt.plot(oil_hist.index, oil_hist['Close'], label='Crude Oil (WTI)', color='green')

    # Formatting
    plt.title("US Dollar Index (DXY) vs Crude Oil Price (WTI) - Last 3 Months")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)

    # Show plot
    plt.savefig('test.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    fetch_and_plot_dxy_oil()
