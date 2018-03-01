# -*- coding: utf-8 -*-
import logging

class Log_Handler:
    def log_initializer(file_name, disable_log):        
        logger = logging.getLogger(file_name)        
        if not getattr(logger, 'handler_set', None):            
            hdlr = logging.FileHandler(file_name)
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            hdlr.setFormatter(formatter)
            logger.addHandler(hdlr) 
            logger.setLevel(logging.DEBUG)
            logger.disabled = disable_log
    #        logger = logging.getLogger(__name__)
            logger.propagate = False
            logger.handler_set = True
        return logger
    