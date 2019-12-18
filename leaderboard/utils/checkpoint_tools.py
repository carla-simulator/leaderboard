import json
from json import JSONDecodeError
import requests


def fetch_dict(endpoint):
    data = None
    if endpoint.startswith(('http:', 'https:', 'ftp:')):
        r = requests.get(url=endpoint)
        data = r.json()
    else:
        with open(endpoint) as fd:
            try:
                data = json.load(fd)
            except JSONDecodeError:
                data = {}

    return data

def create_default_json_msg():
    msg = {
        "status": "ok",
        "value": {
            "results": {
                "values": [],
                "_checkpoint": {
                    "progress": [],
                    "records": [],
                    "global_record": {}
                },
                "is_baseline": 'true',
                "filtering_error": 0,
                "filtering_score": 0
            },
            "session_id": ""}
    }

    return msg

def save_dict(endpoint, data):
    if endpoint.startswith(('http:', 'https:', 'ftp:')):
        _ = requests.patch(url=endpoint, data=json.dumps(data))
    else:
        with open(endpoint, 'w') as fd:
            json.dump(data, fd, indent=4, sort_keys=True)