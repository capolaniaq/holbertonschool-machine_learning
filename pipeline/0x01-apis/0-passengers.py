#!/usr/bin/env python3
"""Request the ships with passeger"""

import requests


def availableShips(passengerCount):
    """
    Function that make a get request to StarWars Api ships
    and select someone had a equal or bigger passegers

    Args:
        passengetCount is the integer of passegers to check
    Return:
        list of the ships with had conditions oor list empty
    """
    response = requests.get("https://swapi-api.hbtn.io/api/starships/")
    response = response.json()
    ships = []
    next = None

    while(response['next'] is not None):
        if next is not None:
            response = requests.get(next)
            response = response.json()

        for i, ship in enumerate(response['results']):
            try:
                if passengerCount <= int(ship['passengers'].replace(',', '')):
                    ships.append(ship['name'])
            except:
                if ship['passengers'] != 'n/a' and \
                            ship['passengers'] != 'unknown':
                    ships.append(ship['name'])
        next = response['next']
    return ships
