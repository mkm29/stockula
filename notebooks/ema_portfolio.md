# Double Exponential Moving Average (34/55) Portfolio Strategy – Performance Addendum

**Prepared for:** Investor\
**Date:** July 26, 2025

______________________________________________________________________

## 1. Overview

This document outlines a 20‑ticker universe and capital allocation plan designed for a 34/55‑period Double Exponential Moving Average (EMA) crossover strategy. The approach balances broad‑market coverage with momentum leaders and a speculative sleeve to capture high‑beta moves.

## 2. Trend‑Tracking Sources Consulted

- Investing.com – Trending Stocks
- Yahoo Finance – Trending Tickers
- MarketWatch – Trending Tickers
- Stocktwits – Sentiment
- StockAnalysis – Trending
- Barchart – Top Trending Tickers
- ApeWisdom – Social Buzz
- Nasdaq.com – Market Activity

## 3. Portfolio Universe (20 Symbols)

| Bucket                                                 | Tickers                                                   | Count |
| ------------------------------------------------------ | --------------------------------------------------------- | ----- |
| Broad‑Market Core SPY, VOO, DFUSX, FSKAX, FSMDX, FXAIX | 6                                                         |       |
| Momentum / Large‑Cap Growth                            | NVDA, AMD, TSM, AAPL, MSFT, GOOGL, AMZN, META, TSLA, PLTR | 10    |
| Speculative High‑Beta                                  | LIDR, OPEN, SOFI, IONQ                                    | 4     |

## 4. Capital Allocation

| Bucket                      | Tickers                                                   | Allocation $ | Allocation % | $ per Ticker |
| --------------------------- | --------------------------------------------------------- | ------------ | ------------ | ------------ |
| Broad‑Market Core           | SPY, VOO, DFUSX, FSKAX, FSMDX, FXAIX                      | $7,000       | 35%          | $1,167       |
| Momentum / Large‑Cap Growth | NVDA, AMD, TSM, AAPL, MSFT, GOOGL, AMZN, META, TSLA, PLTR | $9,500       | 47.5%        | $950         |
| Speculative High‑Beta       | LIDR, OPEN, SOFI, IONQ                                    | $3,500       | 17.5%        | $875         |

## 5. Risk & Execution Notes

- Liquidity: Every symbol trades millions of shares daily; mutual funds execute at daily NAV.
- Signal cadence: Evaluate EMA crossovers on daily bars for ETFs/equities; end‑of‑day for mutual funds.
- Stops: Consider 1‑1.5× ATR for momentum names, 1× ATR for speculative names.
- Rebalance: Review monthly; rotate out tickers that lose trend or liquidity.
- Tax: High turnover in the speculative sleeve may generate short‑term gains.

## 6. Broad‑Market Core 3‑Month Return

Using closing prices from April 25, 2025 through July 25, 2025, the six core index vehicles (SPY, VOO, DFUSX, FSKAX, FSMDX, FXAIX) produced the following simple 3‑month gains:

| Ticker | 3‑Month Return | Source        |
| ------ | -------------- | ------------- |
| SPY    | 15.8%          | Yahoo Finance |
| VOO    | 15.8%          | Yahoo Finance |
| DFUSX  | 14.2%          | Yahoo Finance |
| FSKAX  | 14.9%          | Yahoo Finance |
| FSMDX  | 13.5%          | Yahoo Finance |
| FXAIX  | 15.2%          | Yahoo Finance |

**Average sleeve return:** **14.9%** (arithmetic mean of the six tickers).

______________________________________________________________________

## 7. Monthly and Annualized Sharpe Ratio

For the remaining 14 tickers (momentum and speculative sleeves), the annualized Sharpe ratio requires daily returns. These values were calculated using the `yfinance` Python library (see the `get_data` Jupyter notebook). Below are the results for the April 25 → July 25 2025 price window. Each daily % return was computed, the daily risk‑free rate (based on a 3‑month T‑Bill yield of 4.42%) was subtracted, and the result was annualized using √252.

| Ticker | Annualized Sharpe Ratio | Monthly Sharpe Ratio | Source        |
| ------ | ----------------------- | -------------------- | ------------- |
| NVDA   | 6.15                    | 10.01                | Yahoo Finance |
| MSFT   | 5.57                    | 5.29                 | Yahoo Finance |
| AMD    | 5.40                    | 9.53                 | Yahoo Finance |
| TSM    | 5.29                    | 9.53                 | Yahoo Finance |
| SOFI   | 4.97                    | 5.52                 | Yahoo Finance |
| META   | 3.83                    | 3.50                 | Yahoo Finance |
| OPEN   | 3.36                    | 2.12                 | Yahoo Finance |
| AMZN   | 3.22                    | 11.62                | Yahoo Finance |
| LIDR   | 2.64                    | 3.29                 | Yahoo Finance |
| PLTR   | 2.54                    | 7.24                 | Yahoo Finance |
| GOOGL  | 2.50                    | 7.81                 | Yahoo Finance |
| IONQ   | 2.11                    | 3.12                 | Yahoo Finance |
| TSLA   | 0.71                    | 0.80                 | Yahoo Finance |
| AAPL   | 0.33                    | -0.03                | Yahoo Finance |

______________________________________________________________________

## 8. Next Steps

- Verify that your data vendor matches the same adjusted‑close series used here.
- Consider extending the window to 6 or 12 months for a smoother Sharpe signal.
- After computing Sharpe ratios, you may wish to tilt position sizes toward names with the highest risk‑adjusted returns.
