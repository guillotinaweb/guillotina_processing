from guillotina import configure
from guillotina.catalog.utils import parse_query
from guillotina.interfaces import IResource
from guillotina.utils import find_container
from guillotina.interfaces import IFolder
from guillotina.interfaces import IItem
from guillotina.behaviors.dublincore import IDublinCore

from guillotina_processing.interfaces import ITextExtractor
from guillotina_processing.interfaces import ILabelExtractor

from guillotina_processing.interfaces import IIteratorResources
from guillotina_processing.interfaces import IIteratorSearch
from guillotina_processing.interfaces import IIteratorTypes

from guillotina_processing.utils import get_total_amount_resources
from guillotina_processing.utils import iterate_over_resources

from guillotina_cms.interfaces import IDocument
from guillotina._cache import FACTORY_CACHE

from guillotina.component import query_utility
from guillotina.interfaces import ICatalogUtility

from guillotina.utils import get_object_by_uid


@configure.adapter(
    for_=(IResource),
    provides=IIteratorResources
)
class DeepIterator(object):
    def __init__(self, context):
        self.context = context

    async def deep_walk(self, actual):
        async for obj in actual.async_values():
            if IFolder.providedBy(obj):
                async for children in self.deep_walk(obj):
                    yield children
            elif IItem.providedBy(obj):
                yield obj

    async def total(self):
        count = 0
        for type_name in FACTORY_CACHE.keys():
            count += await get_total_amount_resources(type_name)
        return count

    async def __call__(self, ids=False):
        async for obj in self.deep_walk(self.context):
            if ids:
                yield obj._p_oid
            else:
                yield obj


@configure.adapter(
    for_=(IResource),
    provides=IIteratorTypes
)
class TypeIterator(object):
    def __init__(self, context):
        self.context = context
        self.type = None

    def set_type(self, type_name):
        self.type = type_name

    async def total(self):
        return await get_total_amount_resources(self.type)

    async def __call__(self, ids=False):
        async for obj_id in iterate_over_resources(self.type):
            if ids:
                yield obj_id
            else:
                yield await get_object_by_uid(obj_id)


@configure.adapter(
    for_=(IResource),
    provides=IIteratorSearch
)
class SearchIterator(object):
    def __init__(self, context):
        self.context = context
        self.search = None
        self.container = find_container(self.context)
        self.query = None
        self.utility = query_utility(ICatalogUtility)

    def set_search(self, search):
        if self.container is None:
            raise KeyError('Container needed to be defined')
        self.search = search
        self.parsed_query = parse_query(self.context, self.search, self.utility)

    def set_container(self, container):
        self.container = container

    async def total(self):
        result = await self.utility.search(self.container, self.parsed_query)
        return result['items_count']

    async def __call__(self, ids=False):
        if self.search is None:
            yield []

        if self.container is None:
            yield []

        result = await self.utilty.search(self.container, self.parsed_query)

        from_ = 0

        self.parsed_query['query']['from'] = from_
        result = await self.utility.query(self.context, **self.call_params)
        while len(result['member']) > 0:
            for member in result['member']:
                if ids:
                    yield member['uuid']
                else:
                    obj = await get_object_by_uid(member['uuid'])
                    yield obj
            from_ += len(result['member'])
            self.parsed_query['query']['from'] = from_
            result = await self.utility.search(**self.call_params)


@configure.adapter(
    for_=(IResource),
    provides=ITextExtractor
)
class TextExtractor(object):
    def __init__(self, context):
        self.context = context

    async def __call__(self):
        if self.context.title is not None:
            return self.context.title.split()
        else:
            return []


@configure.adapter(
    for_=(IDocument),
    provides=ITextExtractor
)
class DocumentTextExtractor(object):
    def __init__(self, context):
        self.context = context

    async def __call__(self):
        if self.context.text is not None:
            return self.context.text.data.split()
        else:
            return []


@configure.adapter(
    for_=(IResource),
    provides=ILabelExtractor
)
class DocumentLabelExtractor(object):
    def __init__(self, context):
        self.context = context

    async def __call__(self):
        behavior = IDublinCore(self.context)
        if behavior is None:
            return []
        await behavior.load()
        if behavior.tags is not None:
            return behavior.tags
        else:
            return []
