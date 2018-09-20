import requests, json, \
    urllib.parse # for percent-encoding,
           # i.e convert reserved character to percent-encoding
import os
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Lock
import time

globalLock = Lock()
# api query
queryFormat = \
    'https://www.virustotal.com/ui/search?query={}' # https%3A%2F%2Fwww.baidu.com&

# Input url list, assumed already packed with http
def queryUrl(url, write_path):
    urlQuery = queryFormat.format(urllib.parse.quote(url, safe=''))
    print(urlQuery + ' querying')
    content = (requests.get(urlQuery)).content
    content = json.loads(content)
    sleepTime = 30
    # receive error
    while content.get('error') == None:
        print(urlQuery + 'try again in ', sleepTime)
        time.sleep(sleepTime)
        sleepTime += sleepTime
        content = (requests.get(urlQuery)).content
        content = json.loads(content)

    data = None

    globalLock.acquire()
    with open(write_path, 'r') as f:
        data = json.load(f)
        data.append(json.loads(content))

    with open(write_path, 'w') as f:
        json.dump(data, f)
    globalLock.release()
    time.sleep(5)

def startQuery(url_list, write_path, num_threads):
    # if file is empty initialize it with a pair of []
    if os.path.getsize(write_path) == 0:
        with open(write_path, 'w') as f:
            f.write('[]')

    pool = ThreadPool(num_threads)

    write_path = [write_path] * len(url_list)

    pool.starmap(queryUrl, zip(url_list, write_path))
    pool.close()
    pool.join()

if __name__ == '__main__':
    numThreads = 1
    fileName = 't2.json'

    content = None
    with open('urls.dat', 'r') as f:
        content = f.readlines()
        content = [x.strip() for x in content]

    startQuery(content, fileName, numThreads)
