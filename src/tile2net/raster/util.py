import inspect
import json
import os
import time
from weakref import WeakKeyDictionary

import geopy
import numpy as np
import toolz
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
from toolz import curried, pipe

# import logging
from tile2net.logger import logger


def round_loc(location: list[float], decimals=10) -> list[float]:
    return list(np.around(np.array(location), decimals=decimals))

def southwest_northeast(bbox: list[float]):
    return [
        min(bbox[0], bbox[2]),
        min(bbox[1], bbox[3]),
        max(bbox[0], bbox[2]),
        max(bbox[1], bbox[3]),
    ]

def unpack_relevant(cls, info) -> object:
    if not isinstance(info, dict):
        with open(info) as f:
            kwargs = json.load(f)
    else:
        kwargs = info
    relevant = toolz.keyfilter(
        inspect.signature(cls.__init__).parameters.__contains__,
        kwargs
    )
    res = cls(**relevant)
    return res

class cached_descriptor(property):

    def __init__(self, fget):
        super().__init__(fget)
        self.cache = WeakKeyDictionary()

    # noinspection PyMethodOverriding
    def __get__(self, instance, owner):
        if instance is None:
            return self
        if instance not in self.cache:
            self.cache[instance] = self.fget(instance)
        return self.cache[instance]

    def __set__(self, instance, value):
        self.cache[instance] = value

    def __delete__(self, instance):
        del self.cache[instance]

def geocode(location) -> list[float]:
    if isinstance(location, str):
        try:
            location: list[float] = pipe(
                location.split(','),
                curried.map(float),
                list
            )
        except (ValueError, AttributeError):  # fails if address or list
            # sleep = 10
            # while True:
            #     try:
            #         nom: geopy.Location = Nominatim(user_agent='tile2net').geocode(location)
            #     # except GeocoderTimedOut:
            #     except (
            #             geopy.exc.GeocoderTimedOut,
            #             geopy.exc.GeocoderUnavailable,
            #     ):
            #         logger.info(
            #             f"Geocoding '{location}' timed out, retrying in {sleep} seconds..."
            #         )
            #         time.sleep(sleep)
            #         sleep *= 2
            #     else:
            #         break
            logger.info(f"Geocoding {location}, this may take awhile...")
            nom: geopy.Location = Nominatim(user_agent='tile2net').geocode(location, timeout=None)
            logger.info(f"Geocoded '{location}' to\n\t'{nom.raw['display_name']}'")
            location = pipe(
                nom.raw['boundingbox'],
                # convert lon, lon, lat, lat
                # to lat, lon, lat, lon
                curried.get([0, 2, 1, 3]),
            )
    location = pipe(
        location,
        curried.map(float),
        list,
        round_loc,
        southwest_northeast,
    )
    return location
def name_from_location(location: str | list[float, str]):

    if isinstance(location, str):
        try:
            # location is bbox
            location = pipe(
                location.split(','),
                curried.map(float),
                list,
            )
        except (ValueError, AttributeError):  # fails if already address
            # location is address
            ...
    if isinstance(location, list):
        # location is bbox
        centroid = (
            (location[0] + location[2]) / 2,
            (location[1] + location[3]) / 2,
        )
        logger.info(f"Geocoding {centroid}, this may take awhile...")
        nom: geopy.Location = Nominatim(user_agent='tile2net').reverse(centroid, timeout=None)
        logger.info(f"Geocoded '{centroid}' to\n\t'{nom.raw['display_name']}'")
        location = nom.raw['display_name']
        # geolocator = Nominatim(user_agent='tile2net')
        # location = geolocator.reverse(centroid)
        # sleep = 10
        # while True:
            # try:
            #     location = geolocator.reverse(centroid, )
            # except GeocoderTimedOut:
            #     logger.info(
            #         f"Geocoding '{centroid}' timed out, retrying in {sleep} seconds..."
            #     )
            #     time.sleep(sleep)
            #     sleep *= 2
            # else:
            #     break


    if isinstance(location, str):
        # location is address
        name = pipe(
            location.split(',')[0]
            .replace(' ', '_')
            .casefold(),
            os.path.normcase
        )
        return name
    raise TypeError(f"location must be str or list, not {type(location)}")

if __name__ == '__main__':
    print(name_from_location('New York, NY, USA'))
    print(name_from_location([1.22456789, 2.3456789, 3.456789, 4.56789]))

