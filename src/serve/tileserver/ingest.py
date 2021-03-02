import os, time, json
from datetime import timedelta
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
from tqdm import tqdm
import ee
from shapely import geometry
import rasterio
from src.data import ee_download, create_gt
from src.data.copernicusEMS import activations
from src.data import utils


class Ingestor:
    
    def __init__(self, local_path='tmp', bucket='ml4floods', cloud_path={'S2':'worldfloods/lk-dev/S2','GT':'worldfloods/lk-dev/gt','meta':'worldfloods/lk-dev/meta'}, wait=False):
        """
        An ingestion class to obtain, for a Copernicus EMS event, the S2 imagery and ground truth (GT) floodmap.
        """
        self.local_path = os.path.join(os.getcwd(),local_path) 
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        self.bucket = bucket
        self.cloud_path = cloud_path
        self.wait=wait
        self.logger = logging.getLogger('Ingestor')

        
    def ingest(self,ems_code,code_date):
        """
        Main ingestion pipeline
        """
        
        self.ems_code = ems_code
        self.code_date = code_date
        self.logger.info(f'Ingesting {ems_code}, {code_date}')
        
        # ingest the vector data
        self._ingest_vector()
        
        # ingest the S2 data
        self._ingest_S2_registers()
        
        # build the floodmap raster
        self._construct_GT()
        
        self.logger.info('DONE!')
        
    def _ingest_vector(self):
        
        zip_files_activation = activations.fetch_zip_files(self.ems_code)
        
        # parse into a dataframe to filter aois
        zip_df = pd.DataFrame([
            {
                'z':z,
                'aoi':os.path.split(z)[1].split('_')[1],
                'monprod':os.path.split(z)[1].split('_')[3][0:4],
                'vecrtp':os.path.split(z)[1].split('_')[5][0:3],
            } for z in zip_files_activation
        ])
        
        # prefer product and vectors
        zip_df['pref_monprod'] = zip_df['monprod'].map({'PROD':0,'MONI':1}).fillna(2) # prefer PRODUCT to MONITOR0X to NaN
        zip_df['pref_vecrtp'] = zip_df['vecrtp'].map({'VEC':0,'RTP':1}).fillna(2) # prefer VECTOR to RTP (ready-to-print) to NaN
        
        # dump back to list of paths
        zip_file_paths = zip_df.sort_values(['pref_monprod','pref_vecrtp'])[['aoi','z']].groupby('aoi').nth(0)['z'].values.tolist()

        unzip_files_activation = []
        for zip_path in zip_file_paths:
            aoi_id = os.path.split(zip_path)[1].split('_')[1]
            folder_out = os.path.join(self.local_path,self.ems_code)
            if not os.path.exists(folder_out):
                os.makedirs(folder_out)

            local_zip_file = activations.download_vector_cems(zip_path)
            unzipped_file = activations.unzip_copernicus_ems(local_zip_file, folder_out=folder_out)
            unzip_files_activation.append(unzipped_file)
            self.logger.info(f'unzipped {zip_path}')


        # filter for number of aois

        self.registers = []
        for unzip_folder in unzip_files_activation:
            metadata_floodmap = activations.filter_register_copernicusems(unzip_folder, self.code_date)
            aoi_id = os.path.split(unzip_folder)[1].split('_')[1]
            if metadata_floodmap is not None:
                floodmap = activations.generate_floodmap(metadata_floodmap, folder_files=folder_out)
                self.registers.append({'id':'_'.join([self.ems_code,aoi_id]),"metadata_floodmap": metadata_floodmap, "floodmap": floodmap})
                self.logger.info(f'{unzip_folder} processed successfully.')
            else:
                self.logger.info(f'{unzip_folder} - Error!')
        
    def _ingest_S2_registers(self):
        ### can also into multiprocess...
        try:
            ee.Initialize()
        except:
            raise ValueError('Error initializing EE')
            
        for register in self.registers:
            self.logger.info(f'Ingesting Sentinel-2 {register["metadata_floodmap"]["event id"]} geotiff')
            self._ingest_S2(register)
            
    def _ingest_S2_REST(self,register):
        """Use the REST API..."""
        pass
            
    def _ingest_S2(self,register):
        register_aoi_id = register['metadata_floodmap']['event id'].split('_')[1]
        
        boundary = geometry.box(*register['metadata_floodmap']['area_of_interest_polygon'].bounds)
        ee_boundary = ee.Geometry.Polygon(list(boundary.exterior.coords))
        ee_aoi = ee.Geometry.Polygon(list(register['metadata_floodmap']['area_of_interest_polygon'].exterior.coords))

        date_event = datetime.utcfromtimestamp(register['metadata_floodmap']["satellite date"].timestamp())

        date_end_search = date_event + timedelta(days=20)

        img_col = ee_download.get_s2_collection(date_event, date_end_search, ee_aoi)
        n_images = img_col.size().getInfo()
        imgs_list = img_col.toList(n_images, 0)
        
        self.logger.info(f'For {self.ems_code} {register_aoi_id} found {n_images} images, exporting to bucket')
        
        # export to bucket
        BANDS_EXPORT = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12', 'QA60', 'probability']
        img_export = ee.Image(imgs_list.get(1))
        img_export = img_export.select(BANDS_EXPORT).toFloat().clip(ee_boundary) # .reproject(crs,scale=10).resample('bicubic') resample cannot be used on composites
        
        export_task_fun_img = ee_download.export_task_image(bucket=self.bucket)

        filename = os.path.join(self.cloud_path['S2'],f'{self.ems_code}_{register_aoi_id}_S2')
        desc = os.path.basename(filename)
        task = ee_download.mayberun(filename, desc,
                                    lambda : img_export,
                                    export_task_fun_img,
                                    overwrite=False, dry_run=False,
                                    bucket_name=self.bucket, verbose=2)
        
        if task!=None and self.wait:
            task_status_code = task.status()['state']
            while task_status_code!='COMPLETED':
                self.logger.info(f'task: {task_status_code}, sleeping 10s')
                time.sleep(10)
                task_status_code = task.status()['state']
            print (task.status())
        
    def _construct_GT(self):
        
        self.logger.info('Constructing flood masks')
        
        ### walk through registers
        for register in self.registers:
            
            tiff_address = os.path.join('gs://'+self.bucket,self.cloud_path['S2'],register['id']+'_S2.tif')
            
            # get the floodmask raster for each
            mask = create_gt.compute_water(
                tiffs2=tiff_address, 
                floodmap=register['floodmap'], 
                window=None,
                permanent_water_path=None
            ) 
            
            # grab the crs and transform from the S2 source
            with rasterio.open(tiff_address) as src_s2:
                crs = src_s2.crs
                transform = src_s2.transform
                
            # write the GT mask
            with rasterio.open(
                        os.path.join(self.local_path,f'{register["id"]}_GT.tif'), 
                        'w', 
                        driver='COG', 
                        width=mask.shape[0], 
                        height=mask.shape[1], 
                        count=1,
                        dtype=mask.dtype, 
                        crs=crs, 
                        transform=transform) as dst:
                
                dst.write(mask, indexes=1)
            
            # also save the floodmask_vector.geojson, meta, and aoi.geojson            
            register['floodmap'].to_file(os.path.join(self.local_path,register['id']+'_vector.geojson'),driver='GeoJSON')
            
            ### prep meta for json
            register['metadata_floodmap']['area_of_interest_polygon'] = geometry.mapping(register['metadata_floodmap']['area_of_interest_polygon']) 
            register['metadata_floodmap']['satellite date'] = register['metadata_floodmap']['satellite date'].isoformat()    
            register['metadata_floodmap']['timestamp_pre_event'] = register['metadata_floodmap']['timestamp_pre_event'].isoformat()  
            register['metadata_floodmap']['register_id'] = register['id']

            json.dump(
                register['metadata_floodmap']['area_of_interest_polygon'], 
                open(os.path.join(self.local_path,register['id']+'_aoi.json'),'w')
            )
            print (register['metadata_floodmap'])
            json.dump(
                register['metadata_floodmap'],
                open(os.path.join(self.local_path,register['id']+'_meta.json'),'w')
            )
            
            # move all to cloud
            to_cloud = {
                os.path.join(self.local_path,register['id']+'_vector.geojson'):os.path.join('gs://'+self.bucket,self.cloud_path['meta'],register['id']+'_vector.geojson'),
                os.path.join(self.local_path,register['id']+'_aoi.json'):os.path.join('gs://'+self.bucket,self.cloud_path['meta'],register['id']+'_aoi.json'),
                os.path.join(self.local_path,register['id']+'_meta.json'):os.path.join('gs://'+self.bucket,self.cloud_path['meta'],register['id']+'_meta.json'),
                os.path.join(self.local_path,register['id']+'_GT.tif'):os.path.join('gs://'+self.bucket,self.cloud_path['GT'],register['id']+'_GT.tif'),
            }
            
            for src,dst in to_cloud.items():
                self.logger.info(f'to cloud: {src} -> {dst}')
                utils.save_file_to_bucket(dst, src)
                
            # remove local files
            for kk in to_cloud.keys():
                os.remove(kk)
            
        
        
        
if __name__=="__main__":
    ingestor = Ingestor()
    ingestor.ingest('EMSR501', '2021-01-06')#datetime(2021,1,6,0,0))
    
    
    