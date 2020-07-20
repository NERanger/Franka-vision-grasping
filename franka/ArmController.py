# Copyright (c) 2020 by BionicDL Lab. All Rights Reserved.
# -*- coding:utf-8 -*-
"""
@File: ArmController
@Author: Haokun Wang
@Date: 2020/3/16 15:25
@Description:
"""


class ArmController(object):
    def __init__(self):
        pass

    def move_j(self, *args, **kwargs):
        raise NotImplementedError(' move joint function does not implement. ')

    def move_p(self, *args, **kwargs):
        raise NotImplementedError(' move position function does not implement. ')

    def get_state(self, *args, **kwargs):
        raise NotImplementedError(' get status function does not implement. ')

    def verify_state(self, *args, **kwargs):
        raise NotImplementedError(' check status function does not implement. ')
