# -*- coding: utf-8 -*-

class deeplab_edge():
    def __init__(self,flags):
        self.flags=flags
        self.model=self.get_model()
        
    def get_model(self):
        pass