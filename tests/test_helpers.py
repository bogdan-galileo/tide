import random
from unittest import TestCase

from tests.constants import RANDOM_SEED, TEST_ASSETS_DIR, mAP_threshold
from tidecv.helpers import (create_filtered_Data,
                            enlarge_dataset_to_respect_TIDE,
                            filter_dataset_to_label, json_to_Data)
from tidecv.quantify import TIDE


class TestHelpers(TestCase):
    def setUp(self):
        # Parse json to create the Data structures for GT and Pred
        json_path = f"{TEST_ASSETS_DIR}/soda_df_box_20_images.json"
        self.SODA_gts, self.SODA_preds = json_to_Data(json_path)

        # Call TIDE on the entire Dataset
        # Needed for getting the links and testing the enlarge_dataset method
        self.tide = TIDE(pos_threshold=mAP_threshold)
        self.run = self.tide.evaluate(
            gt=self.SODA_gts, preds=self.SODA_preds, name="tide_run"
        )

    def test_create_filtered_Data(self):
        gts_keep = [0, 5, 10]

        gts_filtered, gts_new_id_to_old_id = create_filtered_Data(
            self.SODA_gts, gts_keep
        )

        # Assert the enlarged ids are what we calculate by hand.
        assert len(gts_filtered.annotations) == 3
        assert gts_new_id_to_old_id == {0: 0, 1: 5, 2: 10}

    def test_enlarge_dataset_to_respect_TIDE_1(self):
        """
        Assert that the added ids are what we can compute manually
        """
        gts_keep = [0, 5, 10]
        preds_keep = [3, 11]

        (
            gts_enlarged,
            preds_enlarged,
            gts_new_id_to_old_id,
            preds_new_id_to_old_id,
        ) = enlarge_dataset_to_respect_TIDE(
            self.SODA_gts, self.SODA_preds, gts_keep, preds_keep, self.run.errors
        )

        # Assert the enlarged ids are what we calculate by hand.
        assert {gts_new_id_to_old_id[gt["_id"]] for gt in gts_enlarged.annotations} == {
            0,
            5,
            10,
            17,
            24,
        }
        assert {
            preds_new_id_to_old_id[pred["_id"]] for pred in preds_enlarged.annotations
        } == {3, 11, 20, 24, 28}

    def test_enlarge_dataset_to_respect_TIDE_2(self):
        """
        Assert that if we re-run TIDE, we do have the same error types
        """
        # Select any 50 preds and 50 gts to start with.
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

        tide_filtered = TIDE(pos_threshold=mAP_threshold)
        run_filtered = tide_filtered.evaluate(
            gt=gts_enlarged, preds=preds_enlarged, name="tide_run"
        )

        # Assert that the errors in the filtered dataset are exaclty
        # the same as the errors in the original.
        # To compare them, we convert the errors into a single uid string
        # Start with the errors in the filtered + enlarged dataset
        filtered_error_uids = set()
        for error in run_filtered.errors:
            error_gt_id = error.gt["id"] if hasattr(error, "gt") else None
            error_pred_id = error.pred["id"] if hasattr(error, "pred") else None
            error_name = error.short_name
            filtered_error_uids.add(f"{error_name}_{error_pred_id}_{error_gt_id}")
        assert len(filtered_error_uids) == len(run_filtered.errors)

        # Now look at the errors in the original dataset (restrict to ids)
        pred_old_ids = set(preds_new_id_to_old_id.values())
        gt_old_ids = set(gts_new_id_to_old_id.values())
        restricted_error_uids = set()
        for error in self.run.errors:
            if (error.is_pred() and error.get_id() in pred_old_ids) or (
                not error.is_pred() and error.get_id() in gt_old_ids
            ):
                error_gt_id = error.gt["id"] if hasattr(error, "gt") else None
                error_pred_id = error.pred["id"] if hasattr(error, "pred") else None
                error_name = error.short_name
                restricted_error_uids.add(f"{error_name}_{error_pred_id}_{error_gt_id}")

        assert filtered_error_uids == restricted_error_uids

    def test_filter_dataset_to_label(self):
        cls_id = 1

        gts_filtered, preds_filtered = filter_dataset_to_label(
            self.SODA_gts, self.SODA_preds, cls_id
        )

        # Assert the filtered dataset is what we calculate by hand.
        assert len(gts_filtered.annotations) == 134
        assert len(preds_filtered.annotations) == 133
