#!/usr/bin/env python3
"""script that prints the location of a specific user"""

import sys
import requests
import time


if __name__ == '__main__':

    url = sys.argv[1]
    response = requests.get(url, headers={'Accept': 'application/vnd.github.v3+json'})
    if response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        init = int(response.headers['X-Ratelimit-Reset'])
        lim = int(time.time())
        X = int((lim - init) / 60)
        print("Reset in {} min".format(X))
    elif response.status_code == 200:
        response = response.json()
        print(response['location'])
