import json, os
from typing import List, Dict, Optional
from requests_html import HTMLSession
import requests
from xml.dom import minidom
from shapely import geometry
import geojson

import pandas as pd
from ml4floods.data.copernicusEMS.activations import *
from ml4floods.data import utils
from google.cloud import storage


COUNTRIES_MAP = {
    'mozambique, eswatini, zimbabwe':'MZ,ZW', 
    'guatemala, honduras, nicaragua':'GT,HN,NI',
    'vietnam':'VN', 
    'moldova, romania, ukraine':'MD,RO,UA', 
    'iran':'IR',
    'french southern...':'TF', 
    'bosnia and herze...':'BA', 
    'united states':'US',
    'honduras, nicaragua':'HN,NI', 
    'guam, northern mariana...':'GU,MP',
    'finland, sweden':'FI,SE', 
    'saint kitts and..., british virgin i...':'KN,VG',
    'puerto rico, u.s. virgin isla...':'PR,VI', 
    'british virgin i...':'VG',
    'netherlands anti...':'AN', 
    'dominican republ..., haiti':'DR,HT',
    'north macedonia':'MK', 
    'bosnia and herze..., serbia':'BA,RS'
}

def postprocess_floodtable(events_df, countries_df):
    events_df = events_df.reset_index()
    countries_df['label_lower'] = countries_df['label'].str.lower()
    events_df['country_lower'] = events_df['Country'].str.lower()
    events_df['CodeDate'] = pd.to_datetime(events_df['CodeDate'])
    events_df = pd.merge(events_df,countries_df, how='left',left_on='country_lower',right_on='label_lower')
    events_df.loc[events_df['value'].isna(),'value'] = events_df.loc[events_df['value'].isna(),'country_lower'].map(COUNTRIES_MAP)
    return events_df
    

def _get_event_aois(ems_code):
    
    url = f'https://emergency.copernicus.eu/mapping/list-of-components/{ems_code}/aemfeed'
    r = requests.get(url)
    tmp_path = os.path.join(os.getcwd(),f'{ems_code}.txt')
    
    with open(tmp_path,'w') as f:
        f.write(r.text)
        
    xmldoc = minidom.parse(tmp_path)
    items = xmldoc.getElementsByTagName('item')
    
    records = []
    for item in items:
        thumbs = item.getElementsByTagName('gdacs:thumbnail')
        polys = item.getElementsByTagName('georss:polygon')

        if len(thumbs)>0:
            t = polys[0].firstChild.data.split(' ') # lat lon
            records.append(
                {
                    'CODE':ems_code,
                    'AOI':os.path.split(thumbs[0].firstChild.data)[1].split('_')[1],
                    'geometry':geometry.mapping(geometry.Polygon([(float(el[0]),float(el[1])) for el in list(zip(t[1::2], t[::2]))])) # lon, lat
                }
            )

    if len(records)>0:
        return pd.DataFrame(records).groupby('AOI').nth(0).reset_index().to_dict(orient='records')
    else: 
        return []

def _ingest_event_aoi(ems_code):
    
    records = _get_event_aois(ems_code)
    for record in records:
        ft = geojson.Feature(geometry=record['geometry'], properties={'CODE':record['CODE'],'AOI':record['AOI']})
        tmp_path = os.path.join(os.getcwd(),'tmp',f'{ems_code}_{record["AOI"]}.json')
        cloud_path = os.path.join('gs://ml4floods','worldfloods','lk-dev','ems-aoi',f'{ems_code}_{record["AOI"]}.json')
        json.dump(ft, open(tmp_path,'w'))
        utils.save_file_to_bucket(cloud_path, tmp_path)
    
def walk_bucket(events_df):
    """
    Walk the google buckets and do two things: 1. ingest any new ems events that are not in the bucket, and 2. add any user-made events to the dataframe.
    if NOT check_available, all this does is add any custom events to the df.
    """
    
    client = storage.Client()
    
    # 1. ingest any new ems blobs
    ems_blobs = [blob.name for blob in client.list_blobs('ml4floods',prefix=os.path.join('worldfloods','lk-dev','ems-aoi')) if blob.name[-5:]=='.json'] # .json -> filter root
    ems_codes = list(set([os.path.split(b)[1].split('_')[0] for b in ems_blobs]))
    do_codes = events_df.loc[~events_df['Code'].isin(ems_codes),'Code'].values.tolist()
    
    print ('walk bucket -> do_codes:', len(do_codes))
    for ems_code in do_codes:
        try:
            _ingest_event_aoi(ems_code)
        except:
            pass
    
    
    # 2. Add custom aois
    custom_blobs = [blob.name for blob in client.list_blobs('ml4floods', prefix=os.path.join('worldfloods','lk-dev','meta')) if 'CUSTOM' in blob.name and blob.name[-5:]=='.json']
    
    print ('walk bucket -> custom blobs:', len(custom_blobs))
    custom_events = []
    for b in custom_blobs:
        blob_json = utils.load_json_from_bucket('ml4floods',os.path.join('worldfloods','lk-dev','meta',os.path.split(b)[1]))
        custom_events.append(
            {
                'Code':os.path.split(b)[1].split('_')[0],
                'Title':os.path.split(b)[1].split('_')[0],
                'CodeDate':blob_json['properties']['event-date'],
                'value':'CUSTOM',
            }
        )
    
    custom_df = pd.DataFrame(custom_events, columns=['Code','Title','CodeDate','value'])
    custom_df['CodeDate'] = pd.to_datetime(custom_df['CodeDate'])
    update_df = events_df.append(custom_df)
    update_df.index = range(len(update_df))

    return update_df

def refresh_geojson(gj_gdf, updated_df):
    """
    Want all events in updated_df to be in geosjon -> 1. add in any custom aois; and 2. any new ems events.
    """
    client = storage.Client()

    # 1. add custom aois
    # get any missing custom events
    missing_custom_events = updated_df.loc[updated_df['Code'].str.contains('CUSTOM') & ~updated_df['Code'].isin(gj_gdf['CODE']),'Code'].values.tolist()
    print ('refresh geojson -> missing_custom_events', missing_custom_events)
    
    #... and add them to the geojson
    missing_blobs = [b.name for b in client.list_blobs('ml4floods', prefix=os.path.join('worldfloods','lk-dev','meta')) if os.path.split(b.name)[1].split('_')[0] in missing_custom_events]
    missing_records = []
    for b in missing_blobs:
        blob_json = utils.load_json_from_bucket('ml4floods',os.path.join('worldfloods','lk-dev','meta',os.path.split(b)[1]))
        missing_records.append(
            {
                'CODE':os.path.split(b)[1].split('_')[0],
                'AOI':os.path.split(b)[1].split('_')[1],
                'TITLE':os.path.split(b)[1].split('_')[0]+'_'+blob_json['properties']['event-date'],
                'event-date':blob_json['properties']['event-date'],
                'geometry':geometry.shape(blob_json['geometry']),
                'value':'CUSTOM'
            }
        )
    
    custom_df = pd.DataFrame(missing_records, columns=['CODE','AOI','TITLE','event-date','geometry','value'])
    custom_df['event-date'] = pd.to_datetime(custom_df['event-date'])
    gj_gdf = gj_gdf.append(gpd.GeoDataFrame(custom_df,geometry='geometry'))
    gj_gdf.index = range(len(gj_gdf))
    
    
    #2. add any new ems events
    # get any missing new ems events
    gj_gdf['idx'] = gj_gdf['CODE']+'_'+gj_gdf['AOI']
    ems_blobs = [blob.name for blob in client.list_blobs('ml4floods',prefix=os.path.join('worldfloods','lk-dev','ems-aoi')) if blob.name[-5:]=='.json'] # .json -> filter root
    
    
    # ... and add them to the geojosn
    ems_blobs = [blob.name for blob in client.list_blobs('ml4floods',prefix=os.path.join('worldfloods','lk-dev','ems-aoi')) if blob.name[-5:]=='.json'] # .json -> filter root
    new_blobs = [b for b in ems_blobs if os.path.split(b)[1].split('_')[0]+'_'+os.path.split(b)[1].split('_')[1][:-5] not in gj_gdf['idx'].values.tolist()]
    
    print ('refresh_gejosn -> new ems_blobs', len(new_blobs))
    #print (new_blobs)
    records = [
        {
            'idx':'_'.join(os.path.split(b)[1].split('_')[0:2]),
            'CODE':os.path.split(b)[1].split('_')[0],
            'AOI':os.path.split(b)[1].split('_')[1][:-5],
            'geometry':geometry.shape(utils.load_json_from_bucket('ml4floods',os.path.join('worldfloods','lk-dev','ems-aoi',os.path.split(b)[1]))['geometry']),
        } 
        for b in new_blobs]
    
    
    new_df = pd.DataFrame(records, columns=['idx','CODE','AOI','geometry'])
    new_df = pd.merge(new_df, updated_df[['Code','CodeDate','Title']], how='left',left_on='CODE',right_on='Code').rename(columns={'CodeDate':'event-date','Title':'TITLE'}).drop(columns=['Code'])
    gj_gdf = gj_gdf.append(gpd.GeoDataFrame(new_df, geometry='geometry'))
    if 'value' in gj_gdf.columns:
        gj_gdf = gj_gdf.drop(columns=['value'])
    gj_gdf = pd.merge(gj_gdf,updated_df[['Code','value']], how='left',left_on='CODE',right_on='Code').drop(columns=['Code'])
    gj_gdf['event-date'] = pd.to_datetime(gj_gdf['event-date'])
    gj_gdf.drop(columns=['idx'])
    gj_gdf.index = range(len(gj_gdf))
    
    return gj_gdf
    
