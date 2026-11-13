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

#Check that the connection is working properly:
print(df.head())