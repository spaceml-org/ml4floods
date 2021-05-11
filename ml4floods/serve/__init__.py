import atexit
import logging
import os
import subprocess
import time
from signal import SIGTERM

import requests

logging.basicConfig(level=logging.INFO)

class TileServer:
    
    def __init__(self):
        self.port=None
        self.server_pid=None
        self.logger=logging.getLogger('TILESERVER')
    
    def serve(self, workers, port):
        atexit.register(self.stop) # make sure the tileserver is stopped if this object dies
        P = subprocess.Popen(['gunicorn', f'--workers={workers}','ml4floods.serve.tileserver.app:app', f'--bind=0.0.0.0:{port}'])
        self.server_pid = P.pid
        self.port=port
        time.sleep(3)  # sleep for 3s to let workers boot
        self.logger.info(f'Started with pid {self.server_pid} port {self.port}')
        
    def stop(self):
        if self.port:
            self.logger.info('Stopping server')
            #r = requests.get('http://localhost:'+str(self.port)+'/terminate')
            os.kill(self.server_pid, SIGTERM)
            self.port=None # turn the port off again
            self.server_pid = None
            time.sleep(3) # sleep for 3s to let working shutdown
            self.logger.info('Server stopped.')
        else:
            self.logger.info('No server running!')
            
            
class ModelServer:
    
    def __init__(self):
        self.port=None
        self.server_pid=None
        self.logger=logging.getLogger('MODELSERVER')
        
    def serve(self, workers, port):
        atexit.register(self.stop) # make sure the tileserver is stopped if this object dies
        P = subprocess.Popen(['gunicorn', f'--workers={workers}','ml4floods.serve.modelserver.app:app', f'--bind=127.0.0.1:{port}'])
        self.server_pid = P.pid
        self.port=port
        time.sleep(3)  # sleep for 3s to let workers boot
        self.logger.info(f'Started with pid {self.server_pid} port {self.port}')
        
    def stop(self):
        if self.port:
            self.logger.info('Stopping server')
            #r = requests.get('http://localhost:'+str(self.port)+'/terminate')
            os.kill(self.server_pid, SIGTERM)
            self.port=None # turn the port off again
            self.server_pid = None
            time.sleep(3) # sleep for 3s to let working shutdown
            self.logger.info('Server stopped.')
        else:
            self.logger.info('No server running!')
            
        
    
