import json, os
from typing import List, Dict, Optional
from requests_html import HTMLSession
import requests
from xml.dom import minidom
from shapely import geometry
import geojson

import pandas as pd
from src.data.copernicusEMS.activations import *
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
    

def walk_bucket(events_df, check_available=True):
    """
    Walk the google buckets and verify the status of the events. The ground truth (GT) and S2 RGB should be available for the image to be 'available', elif there's a token, it is 'pending', else unavailable.    
    """
    
    client = storage.Client()
    s2_blobs = [blob.name for blob in client.list_blobs('ml4floods', prefix=os.path.join('worldfloods','lk-dev','S2'))]
    meta_blobs = [blob.name for blob in client.list_blobs('ml4floods', prefix=os.path.join('worldfloods','lk-dev','meta'))]
    gt_blobs = [blob.name for blob in client.list_blobs('ml4floods', prefix=os.path.join('worldfloods','lk-dev','gt'))]
    ### images: EVENTCODE_AOI_<S2,GT>.tiff
    ### meta: EVENTCODE_AOI.json>
    ### custom: CUSTOM-<utc_timestamp>_AOI_<codedate>_<S2,GT>.<tiff,json>
    
    # if all the tiffs are there, then the 
    s2_recs = [{'PATH':b,'EXT':os.path.splitext(b)[1],'EVENTCODE':os.path.split(b)[1].split('_')[0],'AOI':os.path.split(b)[1].split('_')[1]} for b in s2_blobs if os.path.splitext(b)[1]!='']
    gt_recs = [{'PATH':b,'EXT':os.path.splitext(b)[1],'EVENTCODE':os.path.split(b)[1].split('_')[0],'AOI':os.path.split(b)[1].split('_')[1]} for b in gt_blobs if os.path.splitext(b)[1]!='']
    custom_events = [{'Code':os.path.split(b)[1].split('_')[0],'Title':os.path.split(b)[1].split('_')[0],'CodeDate':os.path.split(b)[1].split('_')[2]} for b in s2_blobs if os.path.split(b)[1][0:6]=='CUSTOM']
    
    custom_df = pd.DataFrame(custom_events)
    custom_df['CodeDate'] = pd.to_datetime(custom_df['CodeDate'])
    update_df = events_df.append(custom_df)
    update_df.index = range(len(update_df))
    
    if not check_available:
        return update_df
    
    s2_df = pd.DataFrame(s2_recs)
    gt_df = pd.DataFrame(gt_recs)
    s2_df['READY'] = s2_df['EXT']=='.tiff'
    gt_df['READY'] = gt_df['EXT']=='.tiff'
    s2_df['PROCESSING'] = s2_df['EXT']=='.token'
    gt_df['PROCESSING'] = gt_df['EXT']=='.token'
    
    # add the READY column -> prod() for AND condition
    update_df = pd.merge(update_df, s2_df[['EVENTCODE','READY']].groupby('EVENTCODE').prod().astype(bool), how='left', left_on='Code',right_index=True).rename(columns={'READY':'S2_READY'})
    update_df = pd.merge(update_df, gt_df[['EVENTCODE','READY']].groupby('EVENTCODE').prod().astype(bool), how='left', left_on='Code',right_index=True).rename(columns={'READY':'GT_READY'})
    update_df['S2_READY'] = update_df['S2_READY'].fillna(0)
    update_df['GT_READY'] = update_df['GT_READY'].fillna(0)
    
    # add the PROCESSING column -> sum() for OR condition
    update_df = pd.merge(update_df, s2_df[['EVENTCODE','PROCESSING']].groupby('EVENTCODE').sum().astype(bool), how='left', left_on='Code',right_index=True).rename(columns={'PROCESSING':'S2_PROCESSING'})
    update_df = pd.merge(update_df, gt_df[['EVENTCODE','PROCESSING']].groupby('EVENTCODE').sum().astype(bool), how='left', left_on='Code',right_index=True).rename(columns={'PROCESSING':'GT_PROCESSING'})
    update_df['S2_READY'] = update_df['S2_READY'].fillna(0)
    update_df['GT_READY'] = update_df['GT_READY'].fillna(0)
    
    return update_df

def _get_event_mp(ems_code):
    
    url = f'https://emergency.copernicus.eu/mapping/list-of-components/{ems_code}/aemfeed'
    r = requests.get(url)
    tmp_path = os.path.join(os.getcwd(),'tmp.txt')
    
    with open(tmp_path,'w') as f:
        f.write(r.text)
        
    xmldoc = minidom.parse(tmp_path)
    items = xmldoc.getElementsByTagName('georss:polygon')
    
    polys = []
    for item in items:
        t = item.firstChild.data.split(' ') # lat lon
        polys.append(geometry.Polygon([(float(el[0]),float(el[1])) for el in list(zip(t[1::2], t[::2]))])) # lon, lat

    os.remove(tmp_path)
    return geometry.MultiPolygon(polys)

def _ingest_event_aoi(ems_code):
    
    multipolygon = _get_event_mp(ems_code)
    ft = geojson.Feature(geometry=multipolygon, properties={'CODE':ems_code,'AOI':None})
    tmp_path = os.path.join(os.getcwd(),'tmp',f'{ems_code}.json')
    cloud_path = os.path.join('gs://ml4floods','worldfloods','lk-dev','ems-aoi',ems_code+'.json')
    json.dump(ft, open(tmp_path,'w'))
    utils.save_file_to_bucket(cloud_path, tmp_path)
    

def refresh_geojson(gj_gdf, updated_df):
    """
    geojson has some mixture of custom, event, and processed aois, updated_df has all events + custom events.
    Want all events in updated_df to be in geosjon. 
    """
    client = storage.Client()

    aoi_blobs = [blob.name for blob in client.list_blobs('ml4floods', prefix=os.path.join('worldfloods','lk-dev','meta')) if blob.name.split('_')[-1]=='aoi.json']
    ems_blobs = [blob.name for blob in client.list_blobs('ml4floods',prefix=os.path.join('worldfloods','lk-dev','ems-aoi')) if blob.name[-5:]=='.json']
    
    # for any codes in updated_df but not in blobs, run grab those aois
    aoi_codes = list(set([os.path.split(b)[1].split('_')[0] for b in aoi_blobs]))
    ems_codes = list(set([os.path.split(b)[1][:-5] for b in ems_blobs]))
    done_codes = list(set([cc for cc in aoi_codes+ems_codes if cc!='']))
    do_codes = updated_df.loc[~updated_df['Code'].isin(done_codes),'Code'].values.tolist()
    do_codes = [cc for cc in do_codes if 'CUSTOM' not in cc]
    print ('do_codes')
    print (do_codes)
    for ems_code in do_codes:
        _ingest_event_aoi(ems_code)
    
    ### supercede any non-ingested ems aois with new aoi_blobs
    # new aoi_codes -> codes that are 'none' in the gdf but are in aoi codes
    new_aoi_codes = set(gj_gdf.loc[gj_gdf['AOI']=='none','CODE'].values.tolist()).intersection(set(aoi_codes))
    print ('interseciton codes')
    print (new_aoi_codes)
    
    # drop these from the gdf
    gj_gdf = gj_gdf.loc[~gj_gdf['CODE'].isin(new_aoi_codes),:]
    
    # ... and add any missing aoi_blobs back in
    new_blobs = [b for b in aoi_blobs if os.path.split(b)[1].split('_')[0] not in gj_gdf['CODE'].values.tolist()]
    print ('new_blobs 1', len(new_blobs))
    records = [
        {
            'CODE':os.path.split(b)[1].split('_')[0],
            'AOI':os.path.split(b)[1].split('_')[1],
            'geometry':geometry.shape(utils.load_json_from_bucket('ml4floods',os.path.join('worldfloods','lk-dev','meta',os.path.split(b)[1]))),
        } 
        for b in new_blobs]
    gj_gdf = gj_gdf.append(gpd.GeoDataFrame(pd.DataFrame(records, columns=['CODE','AOI','geometry']), geometry='geometry'))
    
    # add any missing ems blobs
    new_blobs = [b for b in ems_blobs if os.path.split(b)[1][:-5] not in gj_gdf['CODE'].values.tolist() and 'CUSTOM' not in b]
    print ('new blobs 2')
    print (new_blobs)
    records = [
        {
            'CODE':os.path.split(b)[1][:-5],
            'AOI':'none',
            'geometry':geometry.shape(utils.load_json_from_bucket('ml4floods',os.path.join('worldfloods','lk-dev','ems-aoi',os.path.split(b)[1]))['geometry']),
        } 
        for b in new_blobs]
    gj_gdf = gj_gdf.append(gpd.GeoDataFrame(pd.DataFrame(records, columns=['CODE','AOI','geometry']), geometry='geometry'))
    
    # merge on the date
    for col in ['Title','TITLE','label']:
        if col in gj_gdf.columns:
            gj_gdf =gj_gdf.drop(columns=[col])
    
    gj_gdf = pd.merge(gj_gdf.drop(columns=['event-date']), updated_df[['Code','CodeDate','Title']], how='left',left_on='CODE',right_on='Code').rename(columns={'CodeDate':'event-date','Title':'TITLE'})
    gj_gdf = gj_gdf.drop(columns=['Code'])
    
    return gj_gdf
    