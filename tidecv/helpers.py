import json
from collections import defaultdict
from copy import copy
from typing import List, Tuple

from tidecv.data import Data
from tidecv.errors.error import Error


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
    # Parse the json and convert to Data structure
    for i, ann in enumerate(gt_annotations):
        ann["_id"] = i
        ann["mask"] = None
        ann["ignore"] = False
        ann["class"] = ann["gold"]
        ann["bbox"] = ann["bbox_xywh"]  # boxes already have the required format
        gt_image_ids[ann["image_id"]].append(ann["_id"])

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
    # Parse the json and convert to Data structure
    for i, pred in enumerate(pred_annotations):
        pred["_id"] = i
        pred["mask"] = None
        pred["ignore"] = False
        pred["class"] = pred["pred"]
        pred["score"] = pred["confidence"]
        pred["bbox"] = pred["bbox_xywh"]  # boxes already have the required format
        pred_image_ids[pred["image_id"]].append(pred["_id"])

    preds.annotations = pred_annotations
    # Internal metadata for TIDE, needs to know all the classes
    for i in sorted({pred["class"] for pred in pred_annotations}):
        preds.classes[i] = f"Class {i}"
    # TIDE needs the list of box_ids for every image id in order to do its calculations
    for i, pred in pred_image_ids.items():
        preds.images[i]["name"] = f"Image {i}"
        preds.images[i]["anns"] = pred

    return gts, preds


def create_filtered_Data(
    data: Data,
    ids_keep: set,
    data_name: str = "filtered_data",
    reamapping_new_to_old_ids: dict = None,
) -> Data:
    """
    Create a filtered object Data containing only the annotations with ids in ids_keep
    """
    # Create GTs Data
    data_filtered = Data(name=data_name)

    # Restrict the annotations
    new_id_to_old_id = {}
    annotations = []
    for i, ann in enumerate(
        [ann for ann in data.annotations if ann["_id"] in ids_keep]
    ):
        # We copy the annotation to not change the previous one.
        # TIDE requires all _ids to be of the form range(X), so we re-index and save a dict
        new_id_to_old_id[i] = ann["_id"]
        new_ann = copy(ann)
        new_ann["_id"] = i
        annotations.append(new_ann)

    # Adjust the link for all the preds that have one
    if reamapping_new_to_old_ids is not None:
        reamapping_old_to_new_ids = {
            old: new for new, old in reamapping_new_to_old_ids.items()
        }
        for ann in annotations:
            if "info" in ann and ann["info"].get("matched_with"):
                # The id could be missing from the dict if we don't have
                # all links and the gts is not kept.
                ann["info"]["matched_with"] = reamapping_old_to_new_ids.get(
                    ann["info"]["matched_with"]
                )

    data_filtered.annotations = annotations

    # Restrict the classes
    for i in sorted({ann["class"] for ann in annotations}):
        data_filtered.classes[i] = f"Class {i}"

    # Restrict the images and what annotations they have
    image_ids = defaultdict(list)
    for i, ann in enumerate(annotations):
        image_ids[ann["image_id"]].append(ann["_id"])
    for i, anns in image_ids.items():
        data_filtered.images[i]["name"] = f"Image {i}"
        data_filtered.images[i]["anns"] = anns

    return data_filtered, new_id_to_old_id


def enlarge_dataset_to_respect_TIDE(
    gts: Data, preds: Data, gts_keep: set, preds_keep: set, errors: List[Error]
) -> Tuple[Data, Data, dict, dict]:
    """
    Enlarge completely to respect TIDE, i.e., add all the possible links since
    we want TIDE computed on the filtered dataset = TIDE computed on the large
    dataset + restricted to filtered dataset.

    Adding only direct links pred -> gt is not enough. For example if the
    filtered dataset only contains one Dupe, adding 1-links will add the
    associated GT, but not the pred, so that Dupe becomes a TP in the filtered
    dataset with only 1-links.

    input:
    - gts, preds: the Data instance for gts and preds. The preds instance is
        also used to extract the links pred TP -> gt TP
    - gts_keep, preds_keep: set of ids to keep in the filtered dataset
    - errors: list of errors that is used to extract the links pred -> gt for
        when pred is not a TP

    return:
    - a tuple of Data instances (gts_enlarged, preds_enlarged)
    """

    # Extract a mapping pred error id -> gt assoc id
    pred_id_to_gt_id = {}
    for error in errors:
        if hasattr(error, "pred") and hasattr(error, "gt"):
            pred_id_to_gt_id[error.pred["_id"]] = error.gt["_id"]
    # Add mappings pred TP id -> gt assoc TP id
    pred_id_to_gt_id.update(
        {
            pred["_id"]: pred["info"]["matched_with"]
            for pred in preds.annotations
            if "matched_with" in pred.get("info", {})
        }
    )

    # Add GTs
    assoc_gts = {pred_id_to_gt_id[pred_id] for pred_id in preds_keep}

    # Add Preds
    filetered_gts = set(gts_keep).union(assoc_gts)
    assoc_preds = {
        pred_id for pred_id, gt_id in pred_id_to_gt_id.items() if gt_id in filetered_gts
    }
    filetered_preds = assoc_preds.union(preds_keep)

    # Enlarge them
    gts_enlarged, gts_new_id_to_old_id = create_filtered_Data(gts, filetered_gts)
    preds_enlarged, preds_new_it_to_old_id = create_filtered_Data(
        preds, filetered_preds, reamapping_new_to_old_ids=gts_new_id_to_old_id
    )

    return gts_enlarged, preds_enlarged, gts_new_id_to_old_id, preds_new_it_to_old_id


def filter_dataset_to_label(gts: Data, preds: Data, cls_id: int) -> Tuple[Data, Data]:
    """
    filter a dataset (preds and gts) to only those annotations with a given class.

    input:
    - gtsd, preds: the Data instances for preds and gts
    - cls_id: class to filter by

    return:
    - a tuple of Data instance (gts_filtered, preds_filtered) of ids to keep in the
        filtered dataset
    """

    gts_ids = {gt["_id"] for gt in gts.annotations if gt["class"] == cls_id}
    preds_ids = {pred["_id"] for pred in preds.annotations if pred["class"] == cls_id}

    gts_filtered, gts_new_id_to_old_id = create_filtered_Data(gts, gts_ids)
    preds_filtered, _ = create_filtered_Data(
        preds, preds_ids, reamapping_new_to_old_ids=gts_new_id_to_old_id
    )

    return gts_filtered, preds_filtered
