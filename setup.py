# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-06-13 09:40:02
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-02-21 15:35:56

import os
from setuptools import setup

readme_file = 'README.md'
requirements_file = 'requirements.txt'


def readme():
    ''' Get the content of the README '''
    with open(readme_file, encoding="utf8") as f:
        return f.read()


def get_requirements():
    with open(requirements_file, 'r', encoding="utf8") as f:
        reqs = f.read().splitlines() 
    return reqs


setup(
    name='usnmexps',
    version='1.1',
    description='Code base for the preparation and execution of USNM experiments',
    long_description=readme(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    keywords=('USNM ultrasound neuromodulation'),
    author='Théo Lemaire',
    author_email='theo.lemaire1@gmail.com',
    license='MIT',
    packages=['usnmexps'],
    entry_points={
        'console_scripts': [
            'calibapp=usnmexps.start_calib_app:main',
            'usnmapp=usnmexps.start_usnm_app:main'
        ]
    },
    install_requires=get_requirements(),
    zip_safe=False
)