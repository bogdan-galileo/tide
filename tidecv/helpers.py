import json
from typing import Tuple
from collections import defaultdict
from copy import copy

from tidecv.data import Data


def json_to_Data(json_path: str) -> Tuple[Data, Data]:
    """
    Parse a json file obtained from a vaex dataframe df_box via .to_records()
    and create a GT and Pred structure of type Data which are necessary for
    calling TIDE
    """

    with open(json_path) as jfile:
        data = json.load(jfile)

    # Create GTs Data
    gts = Data(name="test_gt")
    gt_image_ids = defaultdict(list)
    gt_annotations = [ann for ann in data if ann["is_gold"]]
    gt_old_tide_id_to_new_id = {}
    # Parse the json and convert to Data structure
    for i, ann in enumerate(gt_annotations):
        ann["_id"] = i
        # gt_old_tide_id_to_new_id[ann["tide_id"]] = ann["_id"]
        ann["mask"] = None
        ann["ignore"] = False
        ann["class"] = ann["gold"]
        ann["bbox"] = ann["bbox_xywh"] # boxes already have the required format
        gt_image_ids[ann['image_id']].append(ann["_id"])
    
    gts.annotations = gt_annotations
    # Internal metadata for TIDE, needs to know all the classes
    for i in sorted({ann["class"] for ann in gt_annotations}):
        gts.classes[i] = f"Class {i}"
    # TIDE needs the list of box_ids for every image id in order to do its calculations
    for i, anns in gt_image_ids.items():
        gts.images[i]["name"] = f"Image {i}"
        gts.images[i]["anns"] = anns

    # Create Preds
    preds = Data(name="test_pred")
    pred_image_ids = defaultdict(list)
    pred_annotations = [ann for ann in data if ann["is_pred"]]
    pred_old_tide_id_to_new_id = {}
    # Parse the json and convert to Data structure
    for i, pred in enumerate(pred_annotations):
        pred["_id"] = i
        # pred_old_tide_id_to_new_id[ann["tide_id"]] = ann["_id"]
        pred["mask"] = None
        pred["ignore"] = False
        pred["class"] = pred["pred"]
        pred["score"] = pred["confidence"]
        pred["bbox"] = pred["bbox_xywh"] # boxes already have the required format
        pred_image_ids[pred['image_id']].append(pred["_id"])

    preds.annotations = pred_annotations
    # Internal metadata for TIDE, needs to know all the classes
    for i in sorted({pred["class"] for pred in pred_annotations}):
        preds.classes[i] = f"Class {i}"
    # TIDE needs the list of box_ids for every image id in order to do its calculations
    for i, pred in pred_image_ids.items():
        preds.images[i]["name"] = f"Image {i}"
        preds.images[i]["anns"] = pred

    return gts, preds

def create_filtered_Data(data: Data, ids_keep: set, data_name:str = "filtered_data") -> Data:
    """
    Create a filtered object Data containing only the annotations with ids in ids_keep
    """
    # Create GTs Data
    data_filtered = Data(name=data_name)

    # Restrict the annotations
    new_id_to_old_id = {}
    annotations = []
    for i, ann in enumerate([ann for ann in data.annotations if ann["_id"] in ids_keep]):
        # We copy the annotation to not change the previous one.
        # TIDE requires all _ids to be of the form range(X), so we re-index and save a dict
        new_id_to_old_id[i] = ann["_id"]
        new_ann = copy(ann)
        new_ann["_id"] = i
        annotations.append(new_ann)

    data_filtered.annotations = annotations

    # Restrict the classes
    for i in sorted({ann["class"] for ann in annotations}):
        data_filtered.classes[i] = f"Class {i}"
    
    # Restrict the images and what annotations they have
    image_ids = defaultdict(list)
    for i, ann in enumerate(annotations):
        image_ids[ann['image_id']].append(ann["_id"])
    for i, anns in image_ids.items():
        data_filtered.images[i]["name"] = f"Image {i}"
        data_filtered.images[i]["anns"] = anns

    return data_filtered, new_id_to_old_id


def enlarge_dataset_to_respect_TIDE(gts: Data, preds: Data, gts_keep: set, preds_keep: set) -> Tuple[Data, Data, dict, dict]:
    """
    Enlarge completely to respect TIDE, i.e., add all the possible links since we want
    TIDE computed on the filtered dataset = TIDE computed on the large dataset + restricted
    to filtered dataset.

    Adding only direct links pred -> gt is not enough. For example if the filtered
    dataset only contains one Dupe, adding 1-links will add the associated GT, but not the pred,
    so that Dupe becomes a TP in the filtered dataset with only 1-links.

    input:
    - gts, preds: the Data instance for gts and preds
    - gts_keep, preds_keep: set of ids to keep in the filtered dataset

    return:
    - a tuple of Data instances (gts_enlarged, preds_enlarged)
    """

    # Add GTs
    assoc_gts = {pred['info']['matched_with'] for pred in preds.annotations if "matched_with" in pred['info'] and pred["_id"] in preds_keep}
    
    # Add Preds
    filetered_gts = set(gts_keep).union(assoc_gts)
    assoc_preds = {pred['_id'] for pred in preds.annotations if pred['info'].get("matched_with") in filetered_gts}
    filetered_preds = assoc_preds.union(preds_keep)
    
    # Enlarge them
    gts_enlarged, gts_new_id_to_old_id = create_filtered_Data(gts, filetered_gts)
    preds_enlarged, preds_new_it_to_old_id = create_filtered_Data(preds, filetered_preds)

    return gts_enlarged, preds_enlarged, gts_new_id_to_old_id, preds_new_it_to_old_id

def filter_dataset_to_label(gts: Data, preds: Data, cls_id: int) -> Tuple[Data, Data]:
    """
    filter a dataset (preds and gts) to only those annotations with a given class.

    input:
    - gtsd, preds: the Data instances for preds and gts
    - cls_id: class to filter by

    return:
    - a tuple of Data instance (gts_filtered, preds_filtered) of ids to keep in the filtered dataset
    """

    gts_ids = {gt["_id"] for gt in gts.annotations if gt["class"]==cls_id}
    preds_ids = {pred["_id"] for pred in preds.annotations if pred["class"]==cls_id}

    gts_filtered, _ = create_filtered_Data(gts, gts_ids)
    preds_filtered, _ = create_filtered_Data(preds, preds_ids)

    return gts_filtered, preds_filtered