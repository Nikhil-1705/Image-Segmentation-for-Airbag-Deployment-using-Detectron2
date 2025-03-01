from detectron2.data.datasets import register_coco_instances

def register_datasets(dataset, annotations_file_name):
    data_set_name = dataset.name.replace(" ", "-")
    
    # Register train set
    train_data_set_name = f"{data_set_name}-train"
    train_images_dir = os.path.join(dataset.location, "train")
    train_ann_file = os.path.join(train_images_dir, annotations_file_name)
    register_coco_instances(train_data_set_name, {}, train_ann_file, train_images_dir)
    
    # Register test set
    test_data_set_name = f"{data_set_name}-test"
    test_images_dir = os.path.join(dataset.location, "test")
    test_ann_file = os.path.join(test_images_dir, annotations_file_name)
    register_coco_instances(test_data_set_name, {}, test_ann_file, test_images_dir)
    
    # Register validation set
    valid_data_set_name = f"{data_set_name}-valid"
    valid_images_dir = os.path.join(dataset.location, "valid")
    valid_ann_file = os.path.join(valid_images_dir, annotations_file_name)
    register_coco_instances(valid_data_set_name, {}, valid_ann_file, valid_images_dir)
    
    return train_data_set_name, test_data_set_name, valid_data_set_name
