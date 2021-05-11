import os, time, json, requests
from datetime import timedelta
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
from tqdm import tqdm
import ee
from shapely import geometry
import rasterio
from ml4floods.data import ee_download, create_gt
from ml4floods.data.copernicusEMS import activations
from ml4floods.data import utils
from ml4floods.serve.tileserver.REST_mosaic import RESTMosaic


class Ingestor:
    
    def __init__(self, local_path='tmp', bucket='ml4floods', cloud_path={'S2-pre':'worldfloods/lk-dev/S2-pre', 'S2-post':'worldfloods/lk-dev/S2-post','GT':'worldfloods/lk-dev/gt','meta':'worldfloods/lk-dev/meta','ML':'worldfloods/lk-dev/ML'}, include=['S2-pre','S2-post','GT','ML'], workers=1, async_ingest=False, source='REST', inference_endpoint=None, ML_models = ['linear','unet','simplecnn']):
        """
        An ingestion class to obtain, for a Copernicus EMS event, the S2 imagery and ground truth (GT) floodmap.
        """
        self.local_path = os.path.join(os.getcwd(),local_path) 
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        self.bucket = bucket
        self.cloud_path = cloud_path
        self.include=include
        self.source=source  # either 'EE' or 'REST'
        self.async_ingest=False
        self.workers=workers
        self.inference_endpoint=inference_endpoint
        self.BANDS_EXPORT= ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12', 'QA60']#, 'probability']
        self.ML_models=ML_models
        self.logger = logging.getLogger('Ingestor')

        
    def ingest(self,ems_code,code_date,aoi_code, aoi_geom=None):
        """
        Main ingestion pipeline
        """
        
        self.ems_code = ems_code
        self.code_date = code_date
        self.date_obj = datetime.strptime(self.code_date,'%Y-%m-%d')
        self.aoi_code = aoi_code
        self.logger.info(f'Ingesting {ems_code}, {code_date}')
        
        # ingest the vector data
        if aoi_geom!=None and 'GT' in self.include:
            raise ValueError('Specify EITHER an aoi or include "GT"')
        elif 'GT' in self.include:
            # ingest an EMS event
            self._ingest_vector()
        
            # ingest the S2 data
            if self.register==None:
                self.logger.info('Error with vector ingest. Proceeding to ingest S2.')
            self._ingest_S2_register()
            
            self.logger.info('wait a few seconds for the bucket to be ready')
            time.sleep(15)
        
            # build the floodmap raster
            if self.register!=None:
                self._construct_GT()
            
        elif aoi_geom!=None:
            # ingest a custom event
            if 'S2-pre' in self.include:
                save_dest = os.path.join(self.cloud_path['S2-pre'],f'{self.ems_code}_AOI01_S2')
                end_date = self.date_obj - timedelta(days=1)
                start_date = end_date - timedelta(days=21)
                self._ingest_S2_REST(
                    _id='AOI01', 
                    aoi=aoi_geom,
                    start_date=start_date, 
                    end_date=end_date, 
                    save_dest=save_dest)
                
            if 'S2-post' in self.include:
                save_dest = os.path.join(self.cloud_path['S2-post'],f'{self.ems_code}_AOI01_S2')
                end_date = self.date_obj + timedelta(days=20)
                start_date = self.date_obj
                self._ingest_S2_REST(
                    _id='AOI01', 
                    aoi=aoi_geom,
                    start_date=start_date, 
                    end_date=end_date, 
                    save_dest=save_dest)
                
        else:
            raise ValueError('Include either the GT or a custom AOI')
            
        if 'ML' in self.include and self.inference_endpoint!=None:
            ### check inference endpoint
            r = requests.get(self.inference_endpoint)
            assert (json.loads(r.text)['status']=='success'), 'Inference server unavailable - check server is active?'
            self.logger.info('Running ML inference')
            
            for chosen_model in self.ML_models:
                self.logger.info(f'Running inference for {chosen_model} model')
                src_path = os.path.join('gs://ml4floods',self.cloud_path['S2-post'],f'{self.ems_code}_{self.aoi_code}_S2.tif')
                dest_path = os.path.join('gs://ml4floods',self.cloud_path['ML'],f'{self.ems_code}_{self.aoi_code}_ML_{chosen_model}.tif')
                print ('chosen_model',chosen_model)
                params = {
                    'tif_source':src_path,
                    'tif_dest':dest_path,
                    'chosen_model':chosen_model,
                }
                r = requests.get('http://127.0.0.1:8001/get/tif_inference',params=params)
                self.logger.info(r.text)
            
        
        self.logger.info('DONE!')
        
    def _ingest_vector(self):


        metadata_floodmap=None
        
        zip_files_activation = activations.fetch_zip_file_urls(self.ems_code)
        
        # parse into a dataframe to filter aois
        zip_df = pd.DataFrame([
            {
                'z':z,
                'aoi':os.path.split(z)[1].split('_')[1],
                'monprod':os.path.split(z)[1].split('_')[3][0:4],
                'vecrtp':os.path.split(z)[1].split('_')[5][0:3],
            } for z in zip_files_activation
        ])
        
        zip_df = zip_df.loc[zip_df['aoi']==self.aoi_code,:]
        
        # prefer product and vectors
        zip_df['pref_monprod'] = zip_df['monprod'].map({'PROD':0,'MONI':1}).fillna(2) # prefer PRODUCT to MONITOR0X to NaN
        zip_df['pref_vecrtp'] = zip_df['vecrtp'].map({'VEC':0,'RTP':1}).fillna(2) # prefer VECTOR to RTP (ready-to-print) to NaN
        
        zip_df = zip_df.sort_values(['pref_monprod','pref_vecrtp'])
        
        for idx, row in zip_df.iterrows():
            # actually keep this a list of paths and loop through.
            zip_file_path = row['z'] # [['aoi','z']].groupby('aoi').nth(0)['z'].values.tolist()

            #unzip_files_activation = []
            #for zip_path in zip_file_paths:
            #    aoi_id = os.path.split(zip_path)[1].split('_')[1]
            folder_out = os.path.join(self.local_path,self.ems_code)
            if not os.path.exists(folder_out):
                os.makedirs(folder_out)

            local_zip_file = activations.download_vector_cems(zip_file_path)
            unzipped_file = activations.unzip_copernicus_ems(local_zip_file, folder_out=folder_out)
            #unzip_files_activation.append(unzipped_file)
            self.logger.info(f'unzipped {zip_file_path}')


            # filter for number of aois

            #self.registers = []
            #for unzip_folder in unzip_files_activation:
            metadata_floodmap = activations.filter_register_copernicusems(unzipped_file, self.code_date)
            self.logger.info(f'retreived metadata_floodmap')
            
            #    aoi_id = os.path.split(unzip_folder)[1].split('_')[1]
            if metadata_floodmap is not None:
                floodmap = activations.generate_floodmap(metadata_floodmap, folder_files=folder_out)
                self.register = {'id':'_'.join([self.ems_code,self.aoi_code]),"metadata_floodmap": metadata_floodmap, "floodmap": floodmap}
                self.logger.info(f'{unzipped_file} processed successfully.')
                break
            else:
                continue
        if metadata_floodmap==None:       
            self.register=None
            self.logger.info(f'{self.ems_code} - Error!')
        
    def _ingest_S2_register(self):
        ### can also into multiprocess...
        try:
            ee.Initialize()
        except:
            raise ValueError('Error initializing EE')
            
        if self.source=='EE':
            ingest_fn = self._ingest_S2_EE
        elif self.source=='REST':
            ingest_fn = self._ingest_S2_REST
            
          
        self.logger.info(f'Ingesting Sentinel-2 {self.aoi_code} geotiff Pre-event')
        if self.register!=None:
            aoi = self.register['metadata_floodmap']['area_of_interest_polygon']
        else:
            self.logger.info('Getting AOI from bucket.')
            aoi = geometry.shape(utils.load_json_from_bucket('ml4floods',os.path.join('worldfloods','lk-dev','ems-aoi',f'{self.ems_code}_{self.aoi_code}.json'))['geometry'])

        if 'S2-pre' in self.include:
            save_dest = os.path.join(self.cloud_path['S2-pre'],f'{self.ems_code}_{self.aoi_code}_S2')
            end_date = self.date_obj - timedelta(days=1)
            start_date = end_date - timedelta(days=21)

            ingest_fn(
                _id=self.aoi_code, 
                aoi=aoi,
                start_date=start_date, 
                end_date=end_date, 
                save_dest=save_dest)

        if 'S2-post' in self.include:
            save_dest = os.path.join(self.cloud_path['S2-post'],f'{self.ems_code}_{self.aoi_code}_S2')
            start_date = self.date_obj
            end_date = start_date + timedelta(days=20)

            ingest_fn(
                _id=self.aoi_code, 
                aoi=aoi,
                start_date=start_date, 
                end_date=end_date, 
                save_dest=save_dest)
            
    def _ingest_S2_REST(self,_id, aoi, start_date, end_date, save_dest):
        """Use the REST API..."""
        cloud_dest = os.path.join('gs://'+self.bucket,save_dest+'.tif')
        mosaicer = RESTMosaic(
            bands=self.BANDS_EXPORT, 
            #s2_tiles_path = os.path.join(os.getcwd(),'assets','s2_tiles.gpkg'), 
            patch_size=1024,
            verbose=True,
            workers=self.workers
        )
        mosaicer.mosaic(aoi,start_date,end_date, cloud_dest=cloud_dest)
            
    def _ingest_S2_EE(self, _id, aoi, start_date, end_date, save_dest):
        
        boundary = geometry.box(*aoi.bounds)
        ee_boundary = ee.Geometry.Polygon(list(boundary.exterior.coords))
        ee_aoi = ee.Geometry.Polygon(list(aoi.exterior.coords))

        img_col = ee_download.get_s2_collection(start_date, end_date, ee_aoi)
        n_images = img_col.size().getInfo()
        imgs_list = img_col.toList(n_images, 0)
        
        self.logger.info(f'For {self.ems_code} {_id} found {n_images} images, exporting to bucket')
        
        # export to bucket

        img_export = ee.Image(imgs_list.get(1))
        img_export = img_export.select(self.BANDS_EXPORT).toFloat().clip(ee_boundary) # .reproject(crs,scale=10).resample('bicubic') resample cannot be used on composites
        
        export_task_fun_img = ee_download.export_task_image(bucket=self.bucket)

        desc = os.path.basename(save_dest)
        task = ee_download.mayberun(save_dest, desc,
                                    lambda : img_export,
                                    export_task_fun_img,
                                    overwrite=False, dry_run=False,
                                    bucket_name=self.bucket, verbose=2)
        
        if task!=None:
            task_status_code = task.status()['state']
            while task_status_code!='COMPLETED':
                self.logger.info(f'task: {task_status_code}, sleeping 10s')
                time.sleep(10)
                task_status_code = task.status()['state']

        
    def _construct_GT(self):
        
        self.logger.info('Constructing flood masks')
        

        tiff_address = os.path.join('gs://'+self.bucket,self.cloud_path['S2-post'],f'{self.ems_code}_{self.aoi_code}_S2.tif')
        self.logger.info('CHECK BUCKET: '+str(utils.check_file_in_bucket_exists('ml4floods',os.path.join(self.cloud_path['S2-post'],f'{self.ems_code}_{self.aoi_code}_S2.tif'))))
        # get the floodmask raster for each
        mask = create_gt.compute_water(
            tiffs2=tiff_address, 
            floodmap=self.register['floodmap'], 
            window=None,
            permanent_water_path=None
        ) 

        # grab the crs and transform from the S2 source
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True, CPL_DEBUG=True):
            with rasterio.open(tiff_address) as src_s2:
                crs = src_s2.crs
                transform = src_s2.transform

        # write the GT mask
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True, CPL_DEBUG=True):
            with rasterio.open(
                        os.path.join(self.local_path,f'{self.register["id"]}_GT.tif'), 
                        'w', 
                        driver='COG', 
                        width=mask.shape[1], 
                        height=mask.shape[0], 
                        count=1,
                        dtype=mask.dtype, 
                        crs=crs, 
                        transform=transform) as dst:

                dst.write(mask, indexes=1)

        # also save the floodmask_vector.geojson, meta, and aoi.geojson            
        self.register['floodmap'].to_file(os.path.join(self.local_path,self.register['id']+'_vector.geojson'),driver='GeoJSON')

        ### prep meta for json
        self.register['metadata_floodmap']['area_of_interest_polygon'] = geometry.mapping(self.register['metadata_floodmap']['area_of_interest_polygon']) 
        self.register['metadata_floodmap']['satellite date'] = self.register['metadata_floodmap']['satellite date'].isoformat()    
        self.register['metadata_floodmap']['timestamp_pre_event'] = self.register['metadata_floodmap']['timestamp_pre_event'].isoformat()  
        self.register['metadata_floodmap']['register_id'] = self.register['id']

        json.dump(
            self.register['metadata_floodmap']['area_of_interest_polygon'], 
            open(os.path.join(self.local_path,self.register['id']+'_aoi.json'),'w')
        )
        json.dump(
            self.register['metadata_floodmap'],
            open(os.path.join(self.local_path,self.register['id']+'_meta.json'),'w')
        )

        # move all to cloud
        to_cloud = {
            os.path.join(self.local_path,self.register['id']+'_vector.geojson'):os.path.join('gs://'+self.bucket,self.cloud_path['meta'],self.register['id']+'_vector.geojson'),
            os.path.join(self.local_path,self.register['id']+'_aoi.json'):os.path.join('gs://'+self.bucket,self.cloud_path['meta'],self.register['id']+'_aoi.json'),
            os.path.join(self.local_path,self.register['id']+'_meta.json'):os.path.join('gs://'+self.bucket,self.cloud_path['meta'],self.register['id']+'_meta.json'),
            os.path.join(self.local_path,self.register['id']+'_GT.tif'):os.path.join('gs://'+self.bucket,self.cloud_path['GT'],self.register['id']+'_GT.tif'),
        }

        for src,dst in to_cloud.items():
            self.logger.info(f'to cloud: {src} -> {dst}')
            utils.save_file_to_bucket(dst, src)

        # remove local files
        for kk in to_cloud.keys():
            os.remove(kk)
            
        
        
        
if __name__=="__main__":
    ingestor = Ingestor(
        local_path='tmp',
        bucket='ml4floods',
        cloud_path = {
            'S2-pre':'worldfloods/lk-dev/S2-pre',
            'S2-post':'worldfloods/lk-dev/S2-post',
            'GT':'worldfloods/lk-dev/gt',
            'meta':'worldfloods/lk-dev/meta'
        },
        include=['S2-pre','S2-post','GT','ML'],
        workers=3,
        source='REST'
    )
    ingestor.ingest(ems_code='EMSR411',#'EMSR267',#EMSR255_05MARIENBERG  EMSR255  EMSR411 AOI01
                    aoi_code='AOI01',#'01RUSNE',
                    code_date='2019-11-23', 
                    aoi_geom=None)
    
    
    
