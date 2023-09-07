import os
import json

def file_cache(jsonfile):
    def get_data():
        with open(jsonfile) as f:
            data = json.load(f)
            return data

    def save_data(data):
        with open(jsonfile, 'w') as f:
            json.dump(data, f)

    def decorator(fn):
        def wrapped(*args, **kwargs):
            if os.path.exists(jsonfile):
                print('Getting from cache...')
                return get_data()
            res = fn(*args, **kwargs)
            save_data(res)
            return res
          
        return wrapped
    return decorator


if __name__ == '__main__':
    @file_cache('test-cache.json')
    def get_token():
        # imaginary API call to get token
        import time; time.sleep(3)
        return '7c5d3ca5-0088-49a7-ae30-011931a44075'

    print(get_token())