import psycopg2
import os
import glob
import pandas as pd

class DB:
    def __init__(self):
        self.conn = psycopg2.connect(
                host = "34.171.108.238",
                database="dev-sample",
                user="postgres",
                password="sike")
        self.conn.autocommit = True
        
    def run_query(self, query, fetch = False):
        '''
        Takes select query, returns df.
        '''
        cur = self.conn.cursor()
        try:
            cur.execute(query)
            if fetch:
                df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
                cur.close()
                return df
            else:
                cur.close()
                return 
        except Exception as e:
            print('SQL Query Failed \n')
            print(e)
            return False
    
    def close_connection(self):
        self.conn.close()