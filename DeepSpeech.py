#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from os import environ as ENV
monkey_patch = ENV.get("DS_MPATCH", None)
if monkey_patch:
    print("Loading monkey patch:", monkey_patch)
    exec(open(monkey_patch).read())

if __name__ == '__main__':
    try:
        from deepspeech_training import train as ds_train
    except ImportError:
        print('Training package is not installed. See training documentation.')
        raise

    ds_train.run_script()
