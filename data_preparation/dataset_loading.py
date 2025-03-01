from detectron2.data.datasets import register_coco_instances
from roboflow import Roboflow

def load_dataset(api_key, workspace, project_name, version):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version).download("coco-segmentation")
    return dataset
