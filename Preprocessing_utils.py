"""Auto-generated from Trial.ipynb: ONLY code used in cleaning cells, wrapped as steps."""
import pandas as pd
    
def load_data(path = "cairo_real_estate_dataset.csv"):
    df = pd.read_csv(path)
    return df

def indexing(df):
    df.set_index("listing_id",inplace = True)
    return df
def handle_wrong_data(df):
    df = df[(df["bedrooms"] == 2) | (df["bedrooms"] == 3)]
    df = df[df["view_type"] < 4]
    return df
def remove_duplicates(df):
    df.drop_duplicates(inplace=True)
    return df

def save_data(df,path = r"Data\cairo_real_estate_dataset_cleaned.csv"):
    return df.to_csv(path)

def handle_nulls(df):
    df["compound_name"] = df["compound_name"].fillna("No compound")
    return df

def binary_encoding(df):
    binary_cols = ["has_parking","has_amenities","has_security","has_balcony","is_negotiable"]

    for i in binary_cols:
        df[i] = df[i].map({"No": 0, "Yes": 1})
    df
    return df

def view_encoding(df):
    view_map = {"Street": 1, "Garden": 2, "Compound": 3, "Nile": 4}
    df["view_type"] = df["view_type"].map(view_map)
    df
    return df

def finishing_encoding(df):
    finishing_map = {"Unfinished" : 1 , "Semi-finished" : 2, "Lux" : 3, "Super Lux" : 4}
    df["finishing_type"] = df["finishing_type"].map(finishing_map)
    df
    return df

def drop_unwanted_cols(df):
    df.drop(columns = ["listing_date"],inplace = True)
    df.drop(columns="days_on_market",inplace = True)
    return df



def Iqr(df):
    continous_cols = [
        "price_egp",
        "area_sqm",
        "distance_to_auc_km",
        "distance_to_mall_km",
        "distance_to_metro_km",
    ]

    df_cleaned = df.copy()
    outlier_counts = {col: 0 for col in continous_cols}
    removed_indices = set()

    for compound in df["compound_name"].unique():
        subset_idx = df[df["compound_name"] == compound].index
        subset = df.loc[subset_idx, continous_cols]

        for col in continous_cols:
            Q1 = subset[col].quantile(0.25)
            Q3 = subset[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outlier_mask = (subset[col] < lower) | (subset[col] > upper)
            outlier_counts[col] += outlier_mask.sum()
            removed_indices.update(subset_idx[outlier_mask])

    df.drop(index=list(removed_indices),inplace = True)

    total_removed = len(removed_indices)
    print("Outlier Removal Summary (per compound, IQR method)\n")
    for col, count in outlier_counts.items():
        print(f"{col:<25}: {count} outliers detected")

    print(f"\n🔹 Total unique rows removed: {total_removed}")
    print(f"🔹 Final dataset shape: {df_cleaned.shape}")
    print(f"🔹 Percentage removed: {round((total_removed / len(df)) * 100, 2)}%")
    return df


def preprocess(df):
    df = indexing(df)
    df = handle_nulls(df)
    df = binary_encoding(df)
    df = view_encoding(df)
    df = handle_wrong_data(df)
    df = finishing_encoding(df)
    df = drop_unwanted_cols(df)
    df = Iqr(df)
    df = remove_duplicates(df)
    return df
