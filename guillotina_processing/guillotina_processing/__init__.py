# -*- coding: utf-8 -*-
from guillotina import configure

app_settings = {
    'processing': {
        'max_words_vocab': 50000
    }
}


def includeme(root):
    configure.scan('guillotina_processing.adapter')
    configure.scan('guillotina_processing.vocabulary')
