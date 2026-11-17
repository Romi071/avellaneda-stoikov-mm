import pandas as pd
import numpy as np
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

#Test
print(f"Fills: {n_fills} | Maker Ratio: {maker_percentage:.2f}% | Max DD: ${max_drawdown:.2f} | Edge/Share: ${edge_per_share:.4f}")