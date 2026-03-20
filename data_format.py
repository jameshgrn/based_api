import pandas as pd
import numpy as np


def load_dunne_jerolmack():
    df = pd.read_excel("data/GlobalDatasets.xlsx")
    df = pd.DataFrame({
        'original_source': df['Citation'],
        'site_id': df['Site'],
        'slope': df['Slope'].abs(),
        'width': df['Width (m)'],
        'depth': df['Depth (m)'],
        'discharge': df['Discharge (m3/s)'],
        'source': 'Dunne_Jerolmack',
    })
    # Physical plausibility + remove known overlaps with Deal dataset
    df = df[
        (df['width'] > 0) &
        (df['width'] <= 4000) &
        (df['slope'] >= 1e-5) &
        ~df['original_source'].str.contains('Singer|Li et. al', case=False, na=False)
    ].copy()
    return df


def load_deal():
    df = pd.read_csv("data/HG_data_comp_complete.csv")
    df = df.query("river_class != -1.0").copy()
    df['original_source'] = df['source']
    df['source'] = 'Deal'
    df = df[
        (df['width'] > 0) &
        (df['width'] <= 10000) &
        (df['slope'] > 0) &
        (df['depth'] <= 150)
    ]
    drop_cols = [
        'notes', 'area', 'sed_discharge', 'd90', 'bedload_discharge',
        'erosion_rate', 'velocity', 'd50', 'd84', 'Unnamed: 0',
        'DOI', 'primary_source', 'river_class',
    ]
    return df.drop(columns=drop_cols)


def generate_data():
    deal = load_deal()
    dj = load_dunne_jerolmack()

    keep = ['discharge', 'width', 'depth', 'slope', 'site_id', 'source']
    data = pd.concat([deal[keep], dj[keep]], axis=0)

    numeric = ['discharge', 'width', 'depth', 'slope']
    data[numeric] = data[numeric].apply(pd.to_numeric, errors='coerce')

    valid = (
        data[numeric].notna().all(axis=1) &
        (data[numeric] > 0).all(axis=1) &
        (data['slope'] >= 1e-5) &
        (data['width'] <= 4000) &
        (data['width'] / data['depth'] >= 3) &
        (data['width'] / data['depth'] <= 7000)
    )
    data = data[valid].copy()

    print(f"Total records: {len(data)}")
    print(data['source'].value_counts().to_string())

    data.to_csv('data/based_input_data_clean.csv', index=False)
    return data


if __name__ == '__main__':
    generate_data()
