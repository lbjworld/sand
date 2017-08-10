# coding: utf-8
import os
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def find_latest_model(model_path, model_name):
    """model_name format: <model_name>.<ts>.[params].h5"""
    latest_ts = 0
    latest_path = None
    logger.info('search model_path: {mp}'.format(mp=model_path))
    for root, dirs, files in os.walk(model_path):
        for name in files:
            pieces = name.split('.')
            if pieces[0] == model_name:  # filter by model_name
                if int(pieces[1]) > latest_ts:
                    latest_ts = int(pieces[1])
                    latest_path = os.path.join(root, name)
    return latest_path
