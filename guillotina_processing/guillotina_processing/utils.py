import re
import string
from nltk.corpus import stopwords
from guillotina.component import query_utility
from guillotina.interfaces import ICatalogUtility
from guillotina.utils import get_current_request


REMOVE_PUNC = re.compile('[%s]' % re.escape(string.punctuation))


def cleanup_text(text, language='english'):
    tokens = [REMOVE_PUNC.sub('', w) for w in text]
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words(language))
    tokens = [w for w in tokens if w not in stop_words]
    return tokens


# get the total number of resources

async def get_total_amount_resources(type_name):
    request = get_current_request()
    txn = request._txn
    result = txn.get_total_resources_of_type(type_name)
    return await result[0]['count']


async def iterate_over_resources(type_name, page_size=1000):
    request = get_current_request()
    txn = request._txn
    async for ids in txn._get_page_resources_of_type(
            type_name, page_size=page_size):
        yield ids


async def get_index_values(type, container, max_size=10000):
    utility = query_utility(ICatalogUtility)
    index = await utility.get_container_index_name(container)
    query = {
        "aggregations": {
            "result": {
                "terms": {
                    "field": type,
                    "size": max_size
                }
            }
        }
    }
    result = await utility.conn.search(index=index, body=query, size=0)
    return [
        bucket['key'] for bucket in result['aggregations']['result']['buckets']]

