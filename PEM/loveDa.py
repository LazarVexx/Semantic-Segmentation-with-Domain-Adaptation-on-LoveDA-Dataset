from detectron2.data import DatasetCatalog, MetadataCatalog
import os
from pathlib import Path

def load_loveda_dataset(dataset_dir, split):
    dataset_dicts = []
    image_dir = os.path.join(dataset_dir, split, "images")
    annotation_dir = os.path.join(dataset_dir, split, "annotations")
    
    for image_file in os.listdir(image_dir):
        record = {}
        file_path = os.path.join(image_dir, image_file)
        record["file_name"] = file_path
        record["image_id"] = image_file
        annotation_file = os.path.join(annotation_dir, 
                                     image_file.replace(".jpg", ".png"))
        record["sem_seg_file_name"] = annotation_file
        dataset_dicts.append(record)
    
    return dataset_dicts
def register_loveda():
    dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), 
                              "loveDa")
    for split in ["train", "val"]:
        name = f"loveda_{split}"
        DatasetCatalog.register(name, 
            lambda s=split: load_loveda_dataset(dataset_dir, s))
        
        # Correct way to set metadata
        MetadataCatalog.get(name).set(
            stuff_classes=["background", "building", "road", "water", 
                         "barren", "forest", "agriculture"],
            evaluator_type="sem_seg",
            ignore_label=255,
            stuff_colors=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                         [255, 255, 0], [0, 255, 255], [255, 0, 255]]
        )




# Registra automaticamente il dataset quando il modulo viene importato
register_loveda()
