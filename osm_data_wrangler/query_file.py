from pymongo import MongoClient
from pprint import pprint


def get_db():
    client = MongoClient('mongodb://localhost:27017')
    db = client.sanjose1
    return db

def top_user():
    """
    Top 5 users with most contributions
    """
    return [{
        '$group': {
            '_id': '$created.user',
            'count': {
                '$sum': 1
            }
        }
    }, {
        '$sort': {
            'count': -1
        }
    }, {
        '$limit': 5
    }]

def single_post_users():
    """
    Number of users contributing only once
    """
    return [{
        '$group': {
            '_id': '$created.user',
            'count': {
                '$sum': 1
            }
        }
    }, {
        '$group': {
            '_id': '$count',
            'num_users': {
                '$sum': 1
            }
        }
    }, {
        '$sort': {
            '_id': 1
        }
    }, {
        '$limit': 1
    }]

def most_common_buildings():
    """
    Top 20 building types
    """
    return [{
        '$match': {
            'building': {
                '$exists': 1
            }
        }
    }, {
        '$group': {
            '_id': '$building',
            'count': {
                '$sum': 1
            }
        }
    }, {
        '$sort': {
            'count': -1
        }
    }, {
        '$limit': 20
    }]

def top_amenities():
    '''
    Top 7 amenities
    '''
    return [{
        '$match': {
            'amenity': {
                '$exists': 1
            } 
        } 
    }, {
        '$group': {
            '_id': '$amenity', 
            'count': {
                '$sum':1
            } 
        } 
    }, {
        '$sort': {
            'count':-1
        }
    }, {
        '$limit': 10
    }]

def top_banks():
    '''
    Top five bars
    '''
    return [{
        '$match': {
            'amenity': 
                'bank'
        } 
    }, {
        '$group': {
            '_id': '$name', 
            'count': {
                '$sum':1
            } 
        } 
    }, {
        '$sort': {
            'count':-1
        }
    }, {
        '$limit': 5
    }]

def top_cafes():
    '''
    Top five cafes
    '''
    return [{
        '$match': {
            'amenity': 
                'cafe'
        } 
    }, {
        '$group': {
            '_id': '$name', 
            'count': {
                '$sum':1
            } 
        } 
    }, {
        '$sort': {
            'count':-1
        }
    }, {
        '$limit': 5
    }]

def top_fast_foods():
    '''
    Top five fast food chains
    '''
    return [{
        '$match': {
            'amenity': {
                '$exists': 1
            }, 
            'amenity': 
                'fast_food'
        }
    }, {
        '$group': {
            '_id': '$name', 
            'count': {
                '$sum':1
            } 
        } 
    }, {
        '$sort': {
            'count':-1
        }
    }, {
        '$limit': 5
    }]

def top_restaurants():
    '''
    Top five restaurants
    '''
    return [{
        '$match': {
            'amenity': {
                '$exists': 1
            }, 
            'amenity': 
                'restaurant'
        }
    }, {
        '$group': {
            '_id': '$name', 
            'count': {
                '$sum':1
            } 
        } 
    }, {
        '$sort': {
            'count':-1
        }
    }, {
        '$limit': 5
    }]


if __name__ == '__main__':
    db = get_db()
    print "Total number of nodes: "
    pprint( db.openstreetmaps.find({"type": "node"}).count() )
    
    print 'Total number of ways: '
    pprint( db.openstreetmaps.find({"type": "way"}).count() )
    
    print 'Total number of total documents: '
    pprint( db.openstreetmaps.find().count() )
    

    print "Top 5 contributing users: "
    pprint(list( db.openstreetmaps.aggregate(top_user()) ))


    print "Number of single contributing users: "
    pprint(list( db.openstreetmaps.aggregate(single_post_users()) ))


    print "Top twenty buildings: "
    pprint(list( db.openstreetmaps.aggregate(most_common_buildings()) ))
    

    print "Top ten amenities: "
    pprint(list( db.openstreetmaps.aggregate(top_amenities())))

    print "Top five banks: "
    pprint(list( db.openstreetmaps.aggregate(top_banks())))
    
    print "Top five cafes: "
    pprint(list( db.openstreetmaps.aggregate(top_cafes())))
    

    print "Top five fast food chains: "
    pprint(list( db.openstreetmaps.aggregate(top_fast_foods())))
    
    print "Top five restaurants: "
    pprint(list( db.openstreetmaps.aggregate(top_restaurants())))
    