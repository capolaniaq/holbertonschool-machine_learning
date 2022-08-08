#!/usr/bin/env python3
"""
Update the document
"""

from pymongo import MongoClient


def update_topics(mongo_collection, name, topics):
    """Update the document in topics attribute

    Args:
        mongo_collection that is collection from mongo
        name is the name of school for do a match
        topics value to change
    Return:
        none
    """
    query = { 'name' : name }
    new_values = {'$set': {'topics' : topics} }
    mongo_collection.update_many(query, new_values)
