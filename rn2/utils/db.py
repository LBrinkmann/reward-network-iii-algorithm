import pymongo
import hashlib
import os


MONGO_URL = os.environ.get('MC_MONGO_URL', "mongodb://localhost:3001/")


# using the combination of environment_id and experimentName, will ensure,
# that we do not upload the same environments twice (in the same experiment)


def create_mongo_env_id(e):
    return hashlib.md5((e['environmentId'] + e['experimentName']).encode()).hexdigest()


def create_mongo_sol_id(e):
    return hashlib.md5((e['environmentId'] + e['modelName']).encode()).hexdigest()


def get_database(db_name='meteor'):
    client = pymongo.MongoClient(MONGO_URL)
    db = client[db_name]
    return db


def _get_player(db):
    col = db["players"]
    return list(col.find())


def _get_solutions(db):
    col = db["solutions"]
    return list(col.find())


def get_collection(db_name, collection_name, query={}):
    db = get_database(db_name)
    col = db[collection_name]
    return list(col.find(query))


def get_solutions(db_name='meteor'):
    db = get_database(db_name)
    solutions = _get_solutions(db)
    player = _get_player(db)
    player_dict = {p['_id']: p['id'] for p in player}
    solutions = [{**s, 'playerId': player_dict.get(s['playerId'])} for s in solutions]
    return solutions


def get_rounds(db_name='meteor'):
    db = get_database(db_name)
    col = db["rounds"]

    return list(col.find())


def get_player_rounds(db_name='meteor'):
    db = get_database(db_name)
    col = db["player_rounds"]

    return list(col.find())


def get_environments(experiment_name, db_name='meteor', collection_name='networks'):
    db = get_database(db_name)
    col = db[collection_name]

    return list(col.find({"experimentName": experiment_name}))


def insert_environments(environments, db_name='meteor', collection_name='networks'):
    db = get_database(db_name)
    col = db[collection_name]

    # environments = [{**e, '_id': create_mongo_env_id(e)} for e in environments]
    inserted_ids = col.insert_many(environments).inserted_ids

    print(f'Inserted a total of {len(inserted_ids)} environments.')


def insert_machine_solutions(solutions, db_name='meteor'):

    db = get_database(db_name)
    col = db["machineSolutions"]

    solutions = [{**s, '_id': create_mongo_sol_id(s)} for s in solutions]
    inserted_ids = col.insert_many(solutions).inserted_ids

    print(f'Inserted a total of {len(inserted_ids)} solutions.')


def insert_solutions(solutions, db_name='meteor'):
    db = get_database(db_name)
    col = db["solutions"]

    inserted_ids = col.insert_many(solutions).inserted_ids

    print(f'Inserted a total of {len(inserted_ids)} solutions.')
