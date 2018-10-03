# -*- encoding: utf-8 -*-
from zope.interface import Interface


class ITextExtractor(Interface):
    """Get text from object"""


class ILabelExtractor(Interface):
    """Get label from object"""


class IIteratorResources(Interface):
    """Walk on objects"""


class IIteratorSearch(Interface):
    """Search objects iterator"""


class IIteratorTypes(Interface):
    """Type iterator"""