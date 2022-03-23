import pandas as pd
import random
import json

def load(file):
    f = open(file)
    users = json.load(f)
    keys = users[0].keys()
    df = {}
    for user in users:
        for key in keys:
            if key not in df:
                df[key] = [user[key]]
            else:
                df[key].append(user[key])
    df = pd.DataFrame.from_dict(df)
    return df


def dump(file_name, di):
    fo = open(file_name, 'w')
    json.dump(di, fo)


def generate_random_users():
    user = {}
    id = ''
    for _ in range(5):
        id += random.choice("abcdefghijklmnopqrstuvwxyz0123456789")
    user["userId"] = id
    # user['bio'] = [bio + random.choice(['this', 'that', 'those', 'these']) for i in range(5)]
    bio = ""
    for i in range(5):
        bio += random.choice(['freelancer', 'foodie', 'travel enthusiast']) + ' '
    bio = bio[:-1]
    user['bio'] = bio
    user["q1"] = random.randint(0, 9)
    user["q2"] = random.randint(0, 9)
    user["q3"] = random.randint(0, 9)
    user["q4"] = random.randint(0, 9)
    user["q5"] = random.randint(0, 9)
    user["q6"] = random.randint(0, 9)
    user["q7"] = random.randint(0, 9)

    return user


dump('random.txt', [generate_random_users() for i in range(100)])
dump('user.txt', generate_random_users())
