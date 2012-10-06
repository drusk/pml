"""
Created on 2012-10-05

@author: drusk
"""

def get_key_of_highest_value(dictionary):
    key_of_highest = None
    highest_val = None
    
    for key in dictionary:
        val = dictionary[key]
        if key_of_highest == None or val > highest_val:
            key_of_highest = key
            highest_val = val

    return key_of_highest
