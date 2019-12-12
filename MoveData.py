#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:33:06 2019

@author: DavidFelipe
"""
import os
import shutil
import random

#classes = ["c1/","c2/","c3"]
#paths = ["./training/", "./validate/", "./test/"]
percents = [0.6,0.2,0.2]

    
class Move_Data:
    def __init__(self, paths, classes, percent=0.2):
        
        self.paths = paths
        self.classes = classes
        self.num_train_elements = []
        self.train_elements = []
        self.percents = percent
        
        for num, class_name in enumerate(classes):
            elements = os.listdir(paths[0] + class_name)
            self.num_train_elements.append(len(elements))
            self.train_elements.append(elements)
            
        print("Num of classes : " + str(len(classes)))
        print("percent distribution : ")
        print(self.percents)
        print(self.num_train_elements)
        
        
    def move_dataset(self):
        
        for num, class_name in enumerate(self.classes):
            path_from = self.paths[0] + class_name
            path_to_1 = self.paths[1] + class_name
            path_to_2 = self.paths[2] + class_name
            """
            For validate
            """
            class_training_elements = self.train_elements[num]
            num_max_move = int(self.percents*len(class_training_elements))
            elements_to_move, left_elements = self.move_random(num_max_move, class_training_elements)
            self.train_elements[num] = left_elements
            for item in elements_to_move:
                shutil.move(path_from + item, path_to_1 + item)
            
            """
            For test
            """
            class_training_elements = left_elements
            elements_to_move, left_elements = self.move_random(num_max_move, class_training_elements)
            self.train_elements[num] = left_elements
            for item in elements_to_move:
                shutil.move(path_from + item, path_to_2 + item)
    
    def move_random(self, max_values, list_elements):
        counter = 0
        elements_extracted = []
        while counter < max_values:
            position = random.randint(0, (len(list_elements) - 1))
            name = list_elements[position]
            elements_extracted.append(name)
            del list_elements[position]
            counter += 1
        print("Reached : " + str(counter))
        return elements_extracted, list_elements
        
        
        
        
        
        
        
        
        