import logging

def setup_logging():
    logging.basicConfig(filename="logs/app.log", level=logging.INFO)
    logging.info("Application started.")