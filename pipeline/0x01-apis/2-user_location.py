#!/usr/bin/env python3
"""script that prints the location of a specific user"""

import requests
import sys
import time


if __name__ == '__main__':
    headers = {'Accept': 'application/vnd.github.v3+json'}
    response = requests.get(sys.argv[1], headers=headers)
    if response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        init = int(response.headers['X-Ratelimit-Reset'])
        lim = int(time.time())
        X = int((lim - init) / 60)
        print(f"Reset in {X} min")
    elif response.status_code == 200:
        response = response.json()
        print(response['location'])
