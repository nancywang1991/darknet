import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets=[('VOC2012', 'train'), ('VOC2012', 'val'), ('VOC2007', 'train'), ('VOC2007', 'val'), ('VOC2007', 'test'),
('a86a4375', 'train'), ('a86a4375', 'val'), ('a86a4375', 'test'), 
('ab2431d9', 'train'), ('ab2431d9', 'val'),('ab2431d9', 'test'),
('a0f66459', 'train'), ('a0f66459', 'val'),('a0f66459', 'test'),
('e70923c4','train'), ('e70923c4','val'),('e70923c4','test'),
('c95c1e82','train'), ('c95c1e82','val'),('c95c1e82','test'),
('cb46fd46','train'), ('cb46fd46','val'),('cb46fd46','test'),
('d7d5f068','train'), ('d7d5f068','val'),('d7d5f068','test'),
('d6532718','train'), ('d6532718','val'),('d6532718','test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "patient"]


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(folder, image_id):
    in_file = open('VOCdevkit/%s/Annotations/%s.xml'%(folder, image_id))
    out_file = open('VOCdevkit/%s/labels/%s.txt'%(folder, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

for folder, image_set in sets:
    if not os.path.exists('VOCdevkit/%s/labels/'%(folder)):
        os.makedirs('VOCdevkit/%s/labels/'%(folder))
    image_ids = open('VOCdevkit/%s/ImageSets/Main/%s.txt'%(folder, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(folder, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/%s/JPEGImages/%s.jpg\n'%(wd, folder, image_id))
        convert_annotation(folder, image_id)
    list_file.close()

