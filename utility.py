import  pandas as pd 

class DataProcessing():
    def __init__(self):
        pass
    def cleaning(self, df, columns=['Ticker', 'Unnamed: 0', 'Repaired?']):
        """
            remove columns
        """
        cols = [col for col in columns if col in list(df)]
        if cols:
            df.drop(columns=cols, inplace=True)
            print("things are")
            return df
        return df
