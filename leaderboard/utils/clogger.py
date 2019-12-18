import json
import logging
import requests

class CLogger():
    """
    TODO:
    """
    def __init__(self, filename=None, endpoint=None):
        self._filename = filename
        self._endpoint = endpoint

        if not filename:
            self._filename = '/tmp/leaderboard.log'

        logging.basicConfig(format='[%(asctime)s] %(message)s',
                            datefmt='%Y%m%d %I:%M:%S %p',
                            filename=self._filename,
                            level=logging.DEBUG)

    def info(self, string):
        logging.info(string)

    def debug(self, string):
        logging.debug(string)

    def error(self, string):
        logging.error(string)

    def warning(self, string):
        logging.warning(string)

    def retrieve_log(self):
        data = ''
        with open(self._filename, 'r') as fd:
            data = fd.read()

        return data

    def post(self, json_data):
        if not self._endpoint:
            self.error('No ENDPOINT assigned.')
            return

        r = requests.post(url=self._endpoint, data =json_data)
        self.info('Post response: {}'.format(r.text))
