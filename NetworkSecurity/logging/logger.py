import logging
import os
from datetime import datetime

Log_file=f"{datetime.now().strftime('%m_%d_%y %H_%M_%S')}.log"


logs_path=os.path.join(os.getcwd(),"logs")

os.makedirs(logs_path,exist_ok=True)

Log_File_Path=os.path.join(logs_path,Log_file)

logging.basicConfig(
    filename=Log_File_Path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)