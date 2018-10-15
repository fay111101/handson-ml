#!/usr/bin/python
# -*- coding: UTF-8 -*-


"""
    File Name: hoursepredict
    Description:
    Author: fay
    date: 26/03/18
    ----------------------------
    Change Activity:
                    26/03/18
    XXX is a module for XXX
    :copyright: (c) Copyright 2017 by Xiuhong Fei.
    :license: iscas, see LICENSE for more details.
"""
__author__ = 'fay'
import os
import tarfile
from six.moves import urllib
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-mlpractice/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()