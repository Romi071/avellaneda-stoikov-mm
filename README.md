# [WIP] avellaneda-stoikov-market-maker

An institutional-style, asynchronous Market Making engine implementing the **Avellaneda-Stoikov (2008)** stochastic optimal control framework (*"High-frequency trading in a limit order book"*). 

Designed as a standalone client architecture connecting over WebSockets and REST to a custom [Limit Order Book](https://github.com/Romi071/limit-order-book) matching engine.

## System Architecture

* **`simulator.py` (Exogenous Market Flow — Completed):** Concurrent `asyncio` & `aiohttp` stochastic flow generator running two decoupled coroutines over a shared connection pool:
  * **Latent Price & Passive Depth (`maker_loop`):** Evolves the unobservable fair price via discrete-time Arithmetic Brownian Motion ($S_{t+\Delta t} = S_t + \sigma \sqrt{\Delta t} Z$) and streams two-sided `GTC` limit orders quantized to a strict $0.01$ tick grid. Includes an $O(1)$ `collections.deque` FIFO garbage collector that actively sweeps stale resting quotes (`POST /cancel`) beyond $N_{\max}$ to prevent heap bloat and ghost-liquidity walls.
  * **Aggressive Order Flow (`taker_loop`):** Models impatient market takers arriving via a Poisson process ($\Delta \tau \sim \text{Exp}(\lambda)$). Fires spread-crossing `IOC` orders sized via a bounded power-law distribution ($\text{Pareto}(\alpha = 1.5)$ in the heavy-tailed Lévy regime) to simulate both retail noise and multi-level institutional liquidity sweeps.
* **`market_maker.py` (The Hot Path — In Progress):** Event-driven `asyncio` execution engine. Consumes live L2 WebSocket depth snapshots, tracks real-time inventory ($q$), and quotes two-sided liquidity around the indifference reservation price:
  $$r(s, q, t) = s - q \gamma \sigma^2 (T - t)$$
  $$\Delta = \gamma \sigma^2 (T - t) + \frac{2}{\gamma} \ln\left(1 + \frac{\gamma}{\kappa}\right)$$
* **`analytics.py` (The Cold Path — Planned):** Post-trade `pandas` & `matplotlib` attribution pipeline. Resamples SQLite execution ticks into OHLCV bars, calibrates rolling volatility ($\sigma$), and decomposes Spread Capture vs. Inventory PnL against a symmetric baseline.

## Simulation Parameters (`simulator.py`)

| Parameter | Symbol | Default Value | Description |
| :--- | :---: | :---: | :--- |
| Initial Fair Price | $S_0$ | `100.00` | Starting latent mid-price |
| ABM Volatility | $\sigma$ | `0.50` | Price diffusion scale per $\sqrt{\text{second}}$ |
| Maker Heartbeat | $\Delta t$ | `0.25s` | Passive quote injection & ABM step interval |
| Max Active Maker Quotes | $N_{\max}$ | `50` | FIFO `deque` threshold for `POST /cancel` garbage collection |
| Taker Arrival Rate | $\lambda$ | `2.0` $\text{s}^{-1}$ | Exponential inter-arrival intensity for `IOC` takers |
| Taker Volume Tail Exponent | $\alpha$ | `1.5` | Capped Pareto shape parameter for heavy-tailed sweep sizing |
