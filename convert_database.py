import numpy as np
import xml.etree.ElementTree as ET
#import matplotlib.pyplot as plt
import os
import pandas as pd

path = 'Data/shadowDb/'

# Converts one xml boundary file to csv format
def convert_xml(img_name):
    tree = ET.parse(path + 'xml/' + img_name + '.xml')
    root = tree.getroot()
    boundary = []
    for child in root:    
        boundary.append([float(child.attrib["x"]), float(child.attrib["y"])])
    return np.array(boundary)

# Converts all boundary files from xml to csv format
def convert_all():
    for img in os.listdir(path + '/img/'):
        img_name = os.path.splitext(img)[0]
        boundary = convert_xml(img_name)
        df = pd.DataFrame(boundary)
        df.to_csv(path + 'csv/' + img_name + '.csv', header=['x', 'y'], index=False)