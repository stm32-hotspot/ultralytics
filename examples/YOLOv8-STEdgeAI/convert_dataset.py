import os
import sys
import json
import shutil
import argparse
from tqdm import tqdm
import yaml

coco_80_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
                    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                    'toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster',
                    'sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

def classes_inspector(non_existing_classes : list=None, 
                      available_classes : list=None) -> None:
    """
    Ensure all defined classes are well present in the dataset

    Args:
        non_existing_classes (list) : list of non found classes from the dataset
        available_classes (list) : list of detected classes in the dataset

    Returns:
        None
    """
    if len(non_existing_classes) > 0:
        print("The following classes were not found: {}".format(non_existing_classes))
        print("Please make sure that your selected classes exist in the following list: {}".format(available_classes))
        print("Exiting the script...")
        sys.exit()
    else:
        print("Converting the dataset ...")

def verify_coco_classes(coco_annotations_file : str=None,
                        classes : list=None) -> None:
    """
    Check if all expected classes are well present in the provided dataset

    Args:
        coco_annotations_file (str) : path to the coco annotation file
        classes (list) : list of the provided classes (from the yaml file)

    Returns:
        None
    """
    print("Analyzing the dataset ...")
    with open(coco_annotations_file, 'r') as f:
        coco_data = json.load(f)

    class_names = set()
    for annotation in tqdm(coco_data['annotations']):
        category_id = annotation['category_id']
        try:
            for category in coco_data['categories']:
                if category['id'] == category_id:
                    class_name = category['name']
                    class_names.add(class_name)
        except Exception as e:
            exceptions = e
    available_classes = list(class_names)
    non_existing_classes = [c for c in classes if c not in available_classes]
    classes_inspector(non_existing_classes, 
                      available_classes)

def convert_coco_to_yolo(coco_annotations_file : str=None, 
                         coco_images_dir : str=None, 
                         classes : list=coco_80_classes, 
                         export_folder : str=None) -> None:
    """
    Core routine that converts coco data to yolo format and exports them

    Args:
        coco_annotations_file (str) : path to the coco annotations directory
        coco_images_dir (str) : path to the images directory
        classes (list) : list of the provided classes (from the yaml file)
        export_folder (str): path converted dataset will be stored

    Returns:
        None
    """     
    verify_coco_classes(coco_annotations_file, 
                        classes)
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)

    export_folder_1 = os.path.join(coco_images_dir, 'labels')
    if not os.path.exists(export_folder_1):
        os.makedirs(export_folder_1)

    export_folder_2 = os.path.join(export_folder_1, 'val')
    if not os.path.exists(export_folder_2):
        os.makedirs(export_folder_2)
    
    with open(coco_annotations_file, 'r') as f:
        coco_data = json.load(f)

    for image_info in tqdm(coco_data['images']):
        image_file_name = image_info['file_name']
        label_file_name = os.path.splitext(image_file_name)[0] + '.txt'
        label_file_path = os.path.join(export_folder, label_file_name)
        label_file_path_2 = os.path.join(export_folder_2, label_file_name)
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image_info['id']:
                try:
                    category_id = annotation['category_id']
                    for category in coco_data['categories']:
                        if category['id'] == category_id:
                            class_name = category['name']
                    if class_name in classes:
                        class_id = classes.index(class_name)
                        x, y, w, h = annotation['bbox']
                        x_center = x + (w / 2)
                        y_center = y + (h / 2)
                        x_center /= image_info['width']
                        y_center /= image_info['height']
                        w /= image_info['width']
                        h /= image_info['height']
                        label_file = open(label_file_path, 'a')
                        label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                        label_file.close()
                        label_file_2 = open(label_file_path_2, 'a')
                        label_file_2.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                        label_file_2.close()
                except Exception as e:
                    exceptions = e



def main():
    parser = argparse.ArgumentParser(description="Convert COCO dataset to YOLO format")
    parser.add_argument('--coco_annotations_file', type=str, required=True, help='Path to COCO annotations JSON file (e.g. --coco_annotations_file /path/to/instances_val2017.json)')
    parser.add_argument('--coco_images_dir', type=str, required=True, help='Path to COCO images directory (e.g. --coco_images_dir /path/to/val2017)')
    parser.add_argument('--classes', type=str, nargs='+', default=coco_80_classes, help='List of classes to include in the conversion (e.g. --classes person bicycle car)')

    args = parser.parse_args()

    coco_annotations_file = args.coco_annotations_file
    coco_images_dir = args.coco_images_dir
    classes = args.classes
    # Create export folder for labels in the same root as images, under 'labels/val'
    images_root = os.path.dirname(coco_images_dir.rstrip('/\\'))
    parent_folder = os.path.dirname(images_root)
    export_folder = coco_images_dir
    
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    convert_coco_to_yolo(
        coco_annotations_file,
        parent_folder,
        classes=classes,
        export_folder=export_folder
    )

    dataset = {
        "train": coco_images_dir,
        "val": coco_images_dir,
        "names": classes,
    }

    with open("dataset.yaml", "w") as f:
        yaml.dump(dataset, f, default_flow_style=False)

if __name__ == "__main__":
    main()

