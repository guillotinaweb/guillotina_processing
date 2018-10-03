# -*- encoding: utf-8 -*-
from guillotina import configure
from guillotina.behaviors.dublincore import IDublinCore
from guillotina.content import Item
from guillotina.interfaces import IItem
from guillotina.schema import Dict
from guillotina.schema import TextLine
from guillotina.schema import Int
import logging

import collections

from guillotina import app_settings

from guillotina_processing.interfaces import IIteratorResources
from guillotina_processing.interfaces import ITextExtractor
from guillotina_processing.interfaces import IIteratorSearch
from guillotina_processing.interfaces import IIteratorTypes
from guillotina_processing.utils import cleanup_text

from guillotina.content import create_content
from guillotina.content import create_content_in_container


logger = logging.getLogger('guillotina_processing')


class IVocabulary(IItem):

    dictionary = Dict(
        title='Vocabulary',
        required=False,
        default={},
        key_type=TextLine(title='word'),
        value_type=Int(title='index')
    )

    reversed_dictionary = Dict(
        title='Reverse Vocabulary',
        required=False,
        default={},
        key_type=Int(title='index'),
        value_type=TextLine(title='word')
    )

    total = Int(title='Total documents processed')


@configure.contenttype(
    type_name='Vocabulary',
    schema=IVocabulary,
    # used for effective and expiration dates to check validity of download
    behaviors=[IDublinCore],
    allowed_types=[])
class Vocabulary(Item):

    async def calculate_reverse(self):
        self.reversed_dictionary = dict(
            zip(self.dictionary.values(), self.dictionary.keys()))


async def build_vocabulary(
        id, context,
        container=None, store_parent=None,
        search=None, type_name=None, language='english',
        max_words=None):
    logger.info('Starting a vocabulary gathering')

    if store_parent:
        vocabulary = await create_content_in_container(
            store_parent, 'Vocabulary', id)
    else:
        vocabulary = await create_content('Vocabulary', id=id)

    if search is None and type_name is None:
        # All content
        iterator = IIteratorResources(context)
        total = await iterator.total()
    elif type_name is not None:
        iterator = IIteratorTypes(context)
        iterator.set_type(type_name)
        total = await iterator.total()
    elif search is not None:
        iterator = IIteratorSearch(context)
        iterator.set_container(container)
        iterator.set_search(search)
        total = await iterator.total()

    count = 0
    dictionary = collections.Counter()
    logger.info('dictionary: total - %d' % total)
    async for resource in iterator():
        count += 1
        text_extractor = ITextExtractor(resource)
        words = await text_extractor()
        words = cleanup_text(words, language)
        if words is not None:
            counting_words = collections.Counter(words)
            dictionary += counting_words
            logger.info(
                'Size : %d Total : %d/%d' % (len(dictionary), count, total))

    if max_words is None:
        max_words = app_settings['processing']['max_words_vocab'] - 1
    most_common = dictionary.most_common(max_words)
    dictionary = dict()
    dictionary["<PAD>"] = 0
    dictionary["<START>"] = 1
    dictionary["<UNK>"] = 2  # unknown
    dictionary["<UNUSED>"] = 3
    for word, _ in most_common:
        dictionary[word] = len(dictionary)
    vocabulary.dictionary = dictionary
    await vocabulary.calculate_reverse()
    vocabulary.total = total
    return vocabulary
