import pandas as pd

def load_and_preprocess(path):
    df = pd.read_csv(path, parse_dates=["Date"], dayfirst=True)
    df = df.groupby(["Date", "Category"]).agg({"Units Sold": "sum"}).reset_index()

    pivot_df = df.pivot(index="Date", columns="Category", values="Units Sold").fillna(0)
    pivot_df = pivot_df.sort_index()
    
    # Already a datetime index, but this line is safe to keep
    pivot_df.index = pd.to_datetime(pivot_df.index)
    
    return pivot_df
