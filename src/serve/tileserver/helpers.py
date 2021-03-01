import json, os
from typing import List, Dict, Optional
from requests_html import HTMLSession

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
    countries_df['label_lower'] = countries_df['label'].str.lower()
    events_df['country_lower'] = events_df['Country'].str.lower()
    events_df['CodeDate'] = pd.to_datetime(events_df['CodeDate'])
    events_df = pd.merge(events_df,countries_df, how='left',left_on='country_lower',right_on='label_lower')
    events_df.loc[events_df['value'].isna(),'value'] = events_df.loc[events_df['value'].isna(),'country_lower'].map(COUNTRIES_MAP)
    return events_df
    

def walk_bucket(events_df):
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
    custom_events = [{'Code':os.path.split(b)[1].split('_')[0],'CodeDate':os.path.split(b)[1].split('_')[2]} for b in s2_blobs if os.path.split(b)[1][0:6]=='CUSTOM']
    
    custom_df = pd.DataFrame(custom_events)
    custom_df['CodeDate'] = pd.to_datetime(custom_df['CodeDate'])
    events_df = events_df.append(custom_df)
    
    s2_df = pd.DataFrame(s2_recs)
    gt_df = pd.DataFrame(gt_recs)
    s2_df['READY'] = s2_df['EXT']=='.tiff'
    gt_df['READY'] = gt_df['EXT']=='.tiff'
    
    events_df = pd.merge(events_df, s2_df[['EVENTCODE','READY']].groupby('EVENTCODE').prod().astype(bool), how='left', left_on='Code',right_index=True).rename(columns={'READY':'S2_READY'})
    events_df = pd.merge(events_df, gt_df[['EVENTCODE','READY']].groupby('EVENTCODE').prod().astype(bool), how='left', left_on='Code',right_index=True).rename(columns={'READY':'GT_READY'})
    events_df['S2_READY'] = events_df['S2_READY'].fillna(0)
    events_df['GT_READY'] = events_df['GT_READY'].fillna(0)
    
    return events_df