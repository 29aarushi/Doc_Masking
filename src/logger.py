# Importing the logging module for logging messages
import logging

# Importing the os module for interacting with the operating system
import os

# Importing datetime for working with dates and times
from datetime import datetime


# Generate a log file name with the current date and time
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create the logs directory path
logs_path=os.path.join(os.getcwd(),"logs")

# Create the logs directory if it doesn't exist
os.makedirs(logs_path,exist_ok=True)

# Create the full log file path
LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

# Configure the logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

