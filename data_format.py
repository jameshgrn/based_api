import pandas as pd
dunne_jerolmack_path = r"data/GlobalDatasets.xlsx"
dunne_jerolmack = pd.read_excel(dunne_jerolmack_path)
dunne_jerolmack = dunne_jerolmack.drop(['tau_*bf', 'D50 (m)'], axis=1)
dunne_jerolmack['bankfull'] = True
dunne_jerolmack.columns = ['source', 'site_id', 'slope', 'width', 'depth', 'discharge', 'bankfull']
dunne_jerolmack['source'] = dunne_jerolmack['source'] .astype(str)
dunne_jerolmack['site_id'] = dunne_jerolmack['source'] .astype(str)
dunne_jerolmack['slope'] = dunne_jerolmack['slope'].abs()
deal_ds_path = "data/HG_data_comp_complete.csv"
deal_ds = pd.read_csv(deal_ds_path)
deal_ds = deal_ds.query("river_class != -1.0")
deal_ds = deal_ds.drop(['notes', 'area', 'sed_discharge', 'd90',
                        'bedload_discharge', 'erosion_rate', 'velocity',
                        'd50', 'd84', 'Unnamed: 0', 'DOI', 'primary_source', 'river_class'], axis=1)
deal_ds['source'] = deal_ds['source'] .astype(str)
deal_ds['site_id'] = deal_ds['source'] .astype(str)
based_input_data = pd.concat([deal_ds, dunne_jerolmack], axis=0)
based_input_data.to_csv('data/based_input_data.csv')
