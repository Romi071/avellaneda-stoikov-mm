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