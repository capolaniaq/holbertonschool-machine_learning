#!/usr/bin/env python3
""""Make requests to SpaceX api"""

import requests


if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches/upcoming'
    response = requests.get(url).json()

    dates = [x['date_unix'] for x in response]

    #min_date = min(dates)

    idx = dates.index(min(dates))
    upcoming = response[idx]

    #for launch in response:
    #    if launch['date_unix'] == min_date:
    #        upcoming = launch

    id_launchpad = upcoming['launchpad']
    url = "https://api.spacexdata.com/v4/launchpads/{}".format(id_launchpad)
    response_launchpad = requests.get(url).json()

    launchpad_name = response_launchpad['name']
    launchpad_locality = response_launchpad['locality']

    id_rocket = upcoming['rocket']
    url = "https://api.spacexdata.com/v4/rockets/{}".format(id_rocket)
    response_rocket = requests.get(url).json()

    rocket_name = response_rocket['name']

    print("{} ({}) {} - {} ({})".format(upcoming['name'],
                                        upcoming['date_local'],
                                        rocket_name, launchpad_name,
                                        launchpad_locality))
