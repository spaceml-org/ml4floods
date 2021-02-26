import time, requests, logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TEST-TILESERVER')

from src.serve import TileServer

logger.info('initialising tileserver object')
server = TileServer()

logger.info('starting tileserver')
server.serve(workers=4,port=8000)

logger.info('testing server')
r = requests.get('http://localhost:8000/null/null/0/0/0.jpg')
print ('status_code',r.status_code)
print ('content',r.content)

logger.info('stopping server')
server.stop()
