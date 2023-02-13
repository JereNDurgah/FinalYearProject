import os
import csv
import xml.etree.ElementTree as ET


def extract_information(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []

    for obj in root.iter('annotation') :

        Classname = getattr(obj.find('object/name'), 'text', None)
        
        xmin = getattr(obj.find('object/bndbox/xmin'), 'text', None)
        
        ymin = getattr(obj.find('object/bndbox/ymin'), 'text', None)
        
        xmax = getattr(obj.find('object/bndbox/xmax'), 'text', None)
        
        ymax = getattr(obj.find('object/bndbox/ymax'), 'text', None)
        
        file_name =getattr(obj.find('filename'), 'text', None)
    
        height = getattr(obj.find('size/height'), 'text', None)
        
        width = getattr(obj.find('size/width'), 'text', None)    
        
        data.append([xml_file,file_name,Classname,width,height, xmin, ymin, xmax, ymax])

    
        return data

def save_to_csv(data, csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['File Path','filename','name','height','width' ,'xmin', 'ymin', 'xmax', 'ymax'])
        writer.writerows(data)

def main():

    Animal='Goat'
    # path to the Pascal VOC dataset directory
    dataset_path = 'PASCAL VOC with XML Files\%s\Annotations'%Animal

    # get a list of all xml files in the dataset directory
    xml_files = [f for f in os.listdir(dataset_path) if f.endswith('.xml')]

    # extract information from each xml file
    data = []
    for xml_file in xml_files:
        data.extend(extract_information(os.path.join(dataset_path, xml_file)))
    
    #Create CSV File
    csv_file = '%s.csv'%Animal

    # save the extracted information to a csv file
    save_to_csv(data, csv_file)

if __name__ == '__main__':
    main()

   