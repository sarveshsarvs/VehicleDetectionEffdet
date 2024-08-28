import os
import shutil
import xml.etree.ElementTree as ET

#paths
images_path = 'path/to/images'
annotations_path = 'path/to/annotations'
output_path = 'path/to/dataset'

#create output directories for each class
classes = set()
for filename in os.listdir(annotations_path):
    if filename.endswith('.xml'):
        tree = ET.parse(os.path.join(annotations_path, filename))
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            classes.add(name)

for cls in classes:
    os.makedirs(os.path.join(output_path, cls), exist_ok=True)

#move images to their respective class directories
for filename in os.listdir(annotations_path):
    if filename.endswith('.xml'):
        tree = ET.parse(os.path.join(annotations_path, filename))
        root = tree.getroot()
        image_name = root.find('filename').text
        image_path = os.path.join(images_path, image_name)
        for obj in root.findall('object'):
            name = obj.find('name').text
            dst = os.path.join(output_path, name, image_name)
            shutil.copy(image_path, dst)
