#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET  # Use cElementTree or lxml if too slow
import re
import pprint
from collections import defaultdict
import codecs
import json

OSM_FILE = "sample_sj_ca.osm"  # Replace this with your osm file
SAMPLE_FILE = "sample_sj_ca.osm"

k = 10 # Parameter: take every k-th top level element

def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag

    Reference:
    http://stackoverflow.com/questions/3095434/
    inserting-newlines-in-xml-file-generated-via-xml-etree-elementtree-in-python
    """
    context = iter(ET.iterparse(osm_file, events=('start', 'end')))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


# with open(SAMPLE_FILE, 'wb') as output:
#     output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
#     output.write('<osm>\n  ')

#     # Write every kth top level element
#     for i, element in enumerate(get_element(OSM_FILE)):
#         if i % k == 0:
#             output.write(ET.tostring(element, encoding='utf-8'))

#     output.write('</osm>')

################################################################################################## 
# Count tags
def count_tags(filename):
    tags = {}
    for event, elem in ET.iterparse(filename):
        if elem.tag in tags:
            tags[elem.tag] += 1
        else:
            tags[elem.tag] = 1
    return tags
print count_tags(OSM_FILE)


################################################################################################## 
# Get the keys
lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

def key_type(element, keys):
    if element.tag == "tag":
        k = element.attrib['k']
        if lower.search(k):
            keys["lower"] += 1
        elif lower_colon.search(k):
            keys["lower_colon"] += 1
        elif problemchars.search(k):
            keys["problemchars"] +=1
        else:
            keys["other"] += 1
    return keys

def process_map(filename):
    keys = {"lower": 0, 
            "lower_colon": 0, 
            "problemchars": 0, 
            "other": 0}
    for event, element in ET.iterparse(filename):
        keys = key_type(element, keys)
    return keys

keys = process_map(OSM_FILE)
pprint.pprint(keys)

################################################################################################## 
# Number of users
def process_map2(filename):
    users = set()
    for _, element in ET.iterparse(filename):
        if 'uid' in element.attrib.keys():
            user = element.attrib['uid']
            users.add(user)
    return users

users = process_map2(OSM_FILE)
print('Unique users: ')
pprint.pprint(users)
print('Now auditing...')

################################################################################################## 
# Audit
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
# post_code_re = re.match(r'^\d{5}$', postcode)

expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
            "Trail", "Parkway", "Commons", "Alley"]

def audit_postcode(invalid_zipcodes, zipcode):
    """ adds invalid postal codes to a dict """
    if not re.match(r'^\d{5}$', zipcode):
        invalid_zipcodes[zipcode] += 1

def is_postcode(elem):
    """ returns postal code"""
    return 'zip' in elem.attrib['k']


def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)

def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")

def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    invalid_zipcodes = defaultdict(int)

    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
                if is_postcode(tag):
                    audit_postcode(invalid_zipcodes, tag.attrib['v'])



    osm_file.close()
    return street_types, invalid_zipcodes

audit(OSM_FILE)

################################################################################################## 
# Build Schema

lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
double_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
address_re = re.compile(r'^addr\:')
street_re = re.compile(r'^street')
ZIPCODE_TAGS = ['addr:postcode', 
                'tiger:zip_left', 
                'tiger:zip_left_1', 
                'tiger:zip_left_2',
                'tiger:zip_left_3', 
                'tiger:zip_left_4', 
                'tiger:zip_right', 
                'tiger:zip_right_1',
                'tiger:zip_right_2', 
                'tiger:zip_right_3', 
                'tiger:zip_right_4']


CREATED = [ "version", "changeset", "timestamp", "user", "uid"]

mapping = {"Ave": "Avenue",
           "Rd": "Road",
           "Blvd": "Boulevard",
           "Cir": "Circle"
            }

def update_name(name, mapping):
    ''' 
    apply mapping transformation to address names
    '''
    try:
        street_name = name.split(' ')
        street_name[-1] = mapping[street_name[-1]]
        return ' '.join(street_name)

    except KeyError:
        mapping['name'] = 'name'
        return name

# process zipcodes and append to result
def process_zipcode(string):
    result = []
    groups = [group.strip() for group in string.split(';')]
    for group in groups:
        if re.match(r'\d{5}\:\d{5}', group):
            group_range = map(int, group.split(':'))
            result += list(map(str, range(group_range[0], group_range[1]+1)))
        elif re.match(r'\d{5}', group):
            result.append(group)
    return result

def shape_element(element):
    '''
    parse through elements for json export
    '''
    node = {}
    address = {}
    pos_attrib = ['lat', 'lon']
    zipcode_tags = ZIPCODE_TAGS

    if element.tag == "node" or element.tag == "way" :
        zipcodes = set()
        # populate tag type
        node['type'] = element.tag

        # parse through attributes
        for attrib in element.attrib:
            if attrib in CREATED:
                if 'created' not in node:
                    node['created'] = {}
                node['created'][attrib] = element.get(attrib)
            elif attrib in pos_attrib:
                continue
            else:
                node[attrib] = element.get(attrib)

        # populate position
        if set(pos_attrib).issubset(element.attrib):
            node['pos'] = [float(element.get('lat')), float(element.get('lon'))]

        # parse second-level tags for nodes
        for child in element:
            # parse second-level tags for ways and populate `node_refs`
            if child.tag == 'nd':
                if 'node_refs' not in node:
                    node['node_refs'] = []
                if 'ref' in child.attrib:
                    node['node_refs'].append(child.get('ref'))

            # throw out not-tag elements and elements without `k` or `v`
            if child.tag != 'tag' or 'k' not in child.attrib or 'v' not in child.attrib:
                continue
            key = child.get('k')
            val = child.get('v')

            # extract zip codes
            if key in zipcode_tags:
                for zipcode in process_zipcode(val):
                    zipcodes.add(zipcode)


            # skip problematic characters
            if re.search(problemchars, key) or re.search(double_colon, key):
                continue

            # parse address
            elif re.search(address_re, key):
                # clean up the street names using mapping dict
                if key == 'addr:street':
                    val = update_name(val, mapping)

                key = key.replace('addr:', '')
                address[key] = val

            # everything else
            else:
                node[key] = val
        
        # add zipcodes field
        if zipcodes:
            node['zipcodes'] = list(zipcodes)


        # compile address
        if len(address) > 0:
            node['address'] = {}
            street_full = None
            street_dict = {}
            street_format = ['prefix', 'name', 'type']
            # parse through address objects
            for key in address:
                val = address[key]
                if re.search(street_re, key):
                    if key == 'street':
                        street_full = val
                    elif 'street:' in key:
                        street_dict[key.replace('street:', '')] = val
                else:
                    node['address'][key] = val
            # assign street or catch-all to compile street dict
            if street_full:
                node['address']['street'] = street_full
            elif len(street_dict) > 0:
                node['address']['street'] = ' '.join([street_dict[key] for key in street_format])
        return node        
        
    else:
        return None
################################################################################################## 
# JSON Export
def process_map3(file_in, pretty = False):
    '''
    main function that initiate the file transformation
    '''
    file_out = "{0}.json".format(file_in)
    data = []
    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return data

process_map3(OSM_FILE, True)