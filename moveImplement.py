#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:26:38 2019

@author: DavidFelipe
"""

import MoveData
classes = ["Clase1/","Clase2/","Clase3/"]
paths = ["./Dataset/training/", "./Dataset/validate/", "./Dataset/test/"]

movedata = MoveData.Move_Data(paths, classes)
movedata.move_dataset()
