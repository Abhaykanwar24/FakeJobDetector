import logging 
import os
import datetime as datetime

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

LOG_FILE = f"{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_file = os.path.join(log_dir, LOG_FILE)


logging.basicConfig(
    filename=log_file,                
    level=logging.INFO,              
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  
    datefmt="%Y-%m-%d %H:%M:%S"       
)
