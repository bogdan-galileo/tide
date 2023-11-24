import random
from collections import defaultdict
from unittest import TestCase

from tests.constants import RANDOM_SEED, TEST_ASSETS_DIR, mAP_threshold
from tidecv.helpers import enlarge_dataset_to_respect_TIDE, json_to_Data
from tidecv.quantify import TIDE


class TestHelpers(TestCase):
    def setUp(self):
        # Parse json to create the Data structures for GT and Pred
        json_path = f"{TEST_ASSETS_DIR}/soda_df_box_20_images.json"
        self.SODA_gts, self.SODA_preds = json_to_Data(json_path)

        # Call TIDE on the entire Dataset
        self.tide = TIDE(pos_threshold=mAP_threshold)
        self.run = self.tide.evaluate(
            gt=self.SODA_gts, preds=self.SODA_preds, name="tide_run"
        )

    def test_recalc_on_filtered(self):
        """
        Test that recalculating mAP/impact on mAP/errors on any filtered
        dataset can be done from the original run tide object (without
        recalculating TIDE from scratch)
        #"""
        # Select any 50 preds and 50 gts.
        random.seed(RANDOM_SEED)
        gts_keep = random.sample([ann["_id"] for ann in self.SODA_gts.annotations], 50)
        preds_keep = random.sample(
            [ann["_id"] for ann in self.SODA_preds.annotations], 50
        )

        (
            gts_enlarged,
            preds_enlarged,
            gts_new_id_to_old_id,
            preds_new_id_to_old_id,
        ) = enlarge_dataset_to_respect_TIDE(
            self.SODA_gts, self.SODA_preds, gts_keep, preds_keep, self.run.errors
        )

        # Calculate TIDE on the filtered (+ enlarged) data
        tide_filtered = TIDE(pos_threshold=mAP_threshold)
        run_filtered = tide_filtered.evaluate(
            gt=gts_enlarged, preds=preds_enlarged, name="tide_run"
        )

        APs_filtered = run_filtered.ap_data.get_APs()
        mAP_filtered = run_filtered.ap_data.get_mAP()
        errors_filtered = tide_filtered.get_main_errors()["tide_run"]

        # Calculate TIDE by restricting to ids
        # First need to get a dict cls -> ids for both GT and Pred
        gts_filtered_cls_to_ids = defaultdict(set)
        for ann in gts_enlarged.annotations:
            gts_filtered_cls_to_ids[ann["class"]].add(gts_new_id_to_old_id[ann["_id"]])
        preds_filtered_cls_to_ids = defaultdict(set)
        for ann in preds_enlarged.annotations:
            preds_filtered_cls_to_ids[ann["class"]].add(
                preds_new_id_to_old_id[ann["_id"]]
            )

        errors_restricted = self.tide.get_main_errors(
            pred_dict=preds_filtered_cls_to_ids, gt_dict=gts_filtered_cls_to_ids
        )["filtered_errors_in_run_tide_run"]
        APs_restricted = self.tide.runs["tide_run"].qualifiers["restricted_APs"]
        mAP_restricted = self.tide.runs["tide_run"].qualifiers["restricted_mAP"]

        # Assert that mAP on filtered/enlarged = restricted TIDE on original
        assert mAP_filtered == mAP_restricted

        # Assert that APs on filtered/enlarged = restricted TIDE on original
        assert APs_filtered == APs_restricted

        # Assert that impact on mAP on filtered/enlarged = restricted TIDE on original
        assert errors_filtered == errors_restricted
        
