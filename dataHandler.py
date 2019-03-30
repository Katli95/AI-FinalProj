import os
import xml.etree.ElementTree as ET

imgDir = "./data/img"
annDir = "./data/annotations"

def read_Imgs():
    all_imgs = []

    for annotationFile in sorted(os.listdir(annDir)):
        img = {'objects': [], 'filename': imgDir + "/" + annotationFile.replace(".xml", ".jpg")}

        tree = ET.parse(annDir + "/" + annotationFile)

        for elem in tree.iter():
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text.lower()
                        img['objects'] += [obj]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        # TODO: REMOVE 'and all(...' ONLY FOR TESTING THE NETWORK ON SPHERES!!  
        if len(img['objects']) > 0 and all(x['name'] == 'sphere' for x in img['objects']):
            all_imgs += [img]

    return all_imgs