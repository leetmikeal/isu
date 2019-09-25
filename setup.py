# -*- coding: utf-8 -*-
import ast
import os
import re

from setuptools import setup

PACKAGE_NAME = 'isu'
SOURCE_DIR = 'isu'

with open(os.path.join(SOURCE_DIR, '__init__.py')) as f:
    match = re.search(r'__version__\s+=\s+(.*)', f.read())
version = str(ast.literal_eval(match.group(1)))


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith('#')]


reqs = parse_requirements('requirements.txt')

setup(
    # metadata
    name=PACKAGE_NAME,
    version=version,
    description='Isu prediction in 3d boxcell data',
    author='Taktpixel Co., Ltd.',
    author_email='tamaki@taktpixel.co.jp',
    url='https://taktpixel.co.jp',

    # options
    packages=[SOURCE_DIR],
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=reqs,
    extras_require={
        'dev': [
            'pytest>=3',
        ],
    },
    entry_points='''
        [console_scripts]
        {app}={src}.{pkg}:main
    '''.format(app=PACKAGE_NAME.replace('_', '-'), src=SOURCE_DIR, pkg=PACKAGE_NAME),
)
