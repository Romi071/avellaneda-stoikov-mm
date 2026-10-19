# avellaneda-stoikov-market-maker [WIP]

An institutional-style, asynchronous Market Making engine implementing the **Avellaneda-Stoikov (2008)** stochastic optimal control framework (*"High-frequency trading in a limit order book"*). 

Designed as a standalone client architecture connecting over WebSockets and REST to a custom [Limit Order Book](https://github.com/Romi071/limit-order-book) matching engine.

## System Architecture

* **`simulator.py` (Exogenous Market Flow):** Evolves a latent Arithmetic Brownian Motion (ABM) fair-value process and injects Poisson-distributed aggressive takers alongside passive L2 background depth.
* **`market_maker.py` (The Hot Path):** Event-driven `asyncio` execution engine. Consumes live L2 WebSocket depth snapshots, tracks real-time inventory ($q$), and quotes two-sided liquidity around the indifference reservation price:
  $$r(s, q, t) = s - q \gamma \sigma^2 (T - t)$$
  $$\Delta = \gamma \sigma^2 (T - t) + \frac{2}{\gamma} \ln\left(1 + \frac{\gamma}{\kappa}\right)$$
* **`analytics.py` (The Cold Path):** Post-trade `pandas` & `matplotlib` attribution pipeline. Resamples SQLite execution ticks into OHLCV bars, calibrates rolling volatility ($\sigma$), and decomposes Spread Capture vs. Inventory PnL against a symmetric baseline.
