# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 12:38:43 2023

@author: pc
"""

from pydantic import BaseModel

class predict(BaseModel):
    text:str