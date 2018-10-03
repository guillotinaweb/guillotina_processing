# -*- coding: utf-8 -*-
from setuptools import find_packages
from setuptools import setup

with open('VERSION', 'r') as f:
    VERSION = f.read().strip('\n')


setup(
    name='guillotina_processing',
    version=VERSION,
    description='Guillotina Processing Helpers',  # noqa
    long_description=(open('README.rst').read() + '\n' +
                      open('CHANGELOG.rst').read()),
    keywords=['asyncio', 'REST', 'Framework', 'transactional', 'machinelearning', 'tensorflow'],
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Internet :: WWW/HTTP',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    url='https://pypi.python.org/pypi/guillotina_processing',
    license='BSD',
    setup_requires=[
        'pytest-runner',
    ],
    zip_safe=True,
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'guillotina_cms',
        'guillotina>=4.2.7',
        'nltk',
        'numpy',
        'keras'
    ]
)
