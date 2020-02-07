import json
from json import JSONDecodeError
import requests
import os.path


def fetch_dict(endpoint):
    data = None
    if endpoint.startswith(('http:', 'https:', 'ftp:')):
        proxies = os.getenv('HTTPS_PROXY', os.getenv('HTTP_PROXY', None))

        if proxies:
            r = requests.get(url=endpoint, proxies=proxies)
        else:
            r = requests.get(url=endpoint)
        data = r.json()
    else:
        data = {}
        if os.path.exists(endpoint):
            with open(endpoint) as fd:
                try:
                    data = json.load(fd)
                except JSONDecodeError:
                    data = {}

    return data

def create_default_json_msg():
    msg = {
            "values": [],
            "_checkpoint": {
                "progress": [],
                "records": [],
                "global_record": {}
                },
            }

    return msg

def save_dict(endpoint, data):
    if endpoint.startswith(('http:', 'https:', 'ftp:')):
        proxies = os.getenv('HTTPS_PROXY', os.getenv('HTTP_PROXY', None))

        if proxies:
            _ = requests.patch(url=endpoint, data=json.dumps(data, indent=4, sort_keys=True), proxies=proxies)
        else:
            _ = requests.patch(url=endpoint, data=json.dumps(data, indent=4, sort_keys=True))
    else:
        with open(endpoint, 'w') as fd:
            json.dump(data, fd, indent=4, sort_keys=True)