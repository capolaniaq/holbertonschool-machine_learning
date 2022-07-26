#!/usr/bin/env python3
"""script that prints the location of a specific user"""

import sys
import requests
import time


if __name__ == '__main__':

    url = sys.argv[1]
    response = requests.get(url)
    if response.status_code == 200:
        response = response.json()
        print(response['location'])
    if response.status_code == 404:
        print("Not found")
    if response.status_code == 403:
        lim = int(response.headers['X-Ratelimit-Reset'])
        init = int(time.time())
        X = int((lim - init) / 60)
        print("Reset in {} min".format(int(X)))

