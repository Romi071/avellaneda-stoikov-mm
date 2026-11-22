import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

#Open a connection to the ledger data base and write the query to only get the Avellaneda-Stoikov market maker trades
db_path = "../Limit Order Book/ledger.db"
connection = sqlite3.connect(db_path)
query = "SELECT * FROM trades WHERE takerTag = 'as_mm' OR makerTag = 'as_mm'"

#Read the SQL query using pandas and close the connection
df = pd.read_sql_query(query, connection)
connection.close()

#Isolate the AS mm buys
buys_mask = ((df['takerTag'] == 'as_mm') & (df['takerSide'] == 'buy')) | ((df['makerTag'] == 'as_mm') & (df['takerSide'] == 'sell'))

#Add a column with the direction of the AS mm operations (+1 for buy, -1 for sell)
df['direction'] = np.where(buys_mask, 1, -1)

#Reconstruct inventory, balance and wealth
df['inventory'] = (df['direction'] * df['amount']).cumsum()
df['balance'] = (-df['direction'] * df['amount'] * df['price']).cumsum()
df['wealth'] = df['balance'] + df['inventory'] * df['price']

#Maximum Drawdown Computation
high_water_mark = df['wealth'].cummax().clip(lower=0)
drawdown = high_water_mark - df['wealth']
max_drawdown = drawdown.max()

#Maker vs Taker fill ratio, obtained by computing the mean on the Boolean mask (True == 1 ; False == 0)
maker_percentage = (df['makerTag'] == 'as_mm').mean() * 100

#Edge per share
n_fills = len(df)
edge_per_share = df['wealth'].iloc[-1] / df['amount'].sum() 

#Calculate Scaled Sharpe Ratio (N * mean / (sqrt(N) * std) = sqrt(N) * mean / std)
pnl_deltas = df['wealth'].diff()
sharpe = np.sqrt(n_fills) * pnl_deltas.mean() / pnl_deltas.std()

# Multi-horizon Volume-Weighted Markout Analysis (Passive Maker Fills)
horizons = [1, 5, 10, 20]
maker_mask = (df['makerTag'] == 'as_mm')
vw_markouts = {}

for k in horizons:
    future_drift = df['direction'] * (df['price'].shift(-k) - df['price'])
    valid = maker_mask & future_drift.notna()
    vw_markouts[k] = round(
        float((future_drift[valid] * df['amount'][valid]).sum() / df['amount'][valid].sum()),
        4,
    )

# Terminal-Marked PnL Attribution (Maker vs. Taker)
terminal_pnl = df['direction'] * df['amount'] * (df['price'].iloc[-1] - df['price'])
maker_pnl = terminal_pnl[maker_mask].sum()
taker_pnl = terminal_pnl[~maker_mask].sum()

#Test
print("Volume-Weighted Maker Markouts ($/share):", vw_markouts)
print(f"Terminal-Marked PnL | Maker: ${maker_pnl:.2f} | Taker: ${taker_pnl:.2f}")

#Plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

ax1.plot(df['wealth'], label='M2M Wealth')
ax1.plot(high_water_mark, linestyle='--', label='High-Water Mark')
ax1.set_ylabel('Wealth ($)')
ax1.legend()

ax2.plot(-drawdown, color='red')
ax2.fill_between(df.index, -drawdown, 0, color='red', alpha=0.2)
ax2.set_ylabel('Drawdown ($)')

ax3.plot(df['inventory'], color='purple')
ax3.axhline(0, color='black', linestyle='--')
ax3.set_ylabel('Inventory (q_t)')
ax3.set_xlabel('Trade Number')

plt.tight_layout()
# plt.show()