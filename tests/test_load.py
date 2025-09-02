import json
import logging
import re
import urllib
import warnings
from pathlib import Path

import geopandas as gpd  # noqa: F401
import numpy as np
import pandas as pd
import pandas.testing as tm
import pyogrio
import pytest
from arcgis import GeoAccessor, GeoSeriesAccessor  # noqa: F401

from palletjack import load


class TestFields:
    def test_get_fields_from_dataframe_returns_all_fields_in_order(self, mocker):
        df_mock = mocker.Mock()
        df_mock.columns = ["Foo", "Bar", "Baz"]

        fields = load.ServiceUpdater._get_fields_from_dataframe(df_mock)

        assert fields == ["Foo", "Bar", "Baz"]

    def test_get_fields_from_dataframe_strips_shape_length(self, mocker):
        df_mock = mocker.Mock()
        df_mock.columns = ["Foo", "Bar", "Baz", "Shape_Length"]

        fields = load.ServiceUpdater._get_fields_from_dataframe(df_mock)

        assert fields == ["Foo", "Bar", "Baz"]

    def test_get_fields_from_dataframe_strips_shape_area(self, mocker):
        df_mock = mocker.Mock()
        df_mock.columns = ["Foo", "Bar", "Baz", "Shape_Area"]

        fields = load.ServiceUpdater._get_fields_from_dataframe(df_mock)

        assert fields == ["Foo", "Bar", "Baz"]

    def test_get_fields_from_dataframe_strips_both_shape_fields(self, mocker):
        df_mock = mocker.Mock()
        df_mock.columns = ["Foo", "Bar", "Baz", "Shape_Length", "Shape_Area"]

        fields = load.ServiceUpdater._get_fields_from_dataframe(df_mock)

        assert fields == ["Foo", "Bar", "Baz"]


class TestServiceUpdaterInit:
    def test_init_calls_get_feature_layer(self, mocker):
        arcgis_mock = mocker.patch("palletjack.load.arcgis")

        updater = load.ServiceUpdater(mocker.Mock(), "itemid")

        arcgis_mock.features.FeatureLayer.fromitem.assert_called_once()

    # #: WIP: removing column name for agol renamer (#36), more __init__ tests
    # def test_init_removes_shape_fields(self, mocker):
    #     new_dataframe = pd.DataFrame(columns=['Foo field', 'Bar', 'Shape_Length', 'Shape_Area'])
    #     mocker.patch('palletjack.load.arcgis')

    #     updater = load.ServiceUpdater(mocker.Mock(), 'itemid', new_dataframe, fields=new_dataframe.columns)

    #     assert set(updater.fields) == {'Foo field', 'Bar'}

    def test_init_sets_working_dir(self, mocker):
        mocker.patch("palletjack.load.arcgis")

        updater = load.ServiceUpdater(mocker.Mock(), "itemid", working_dir=r"c:\foo")

        assert updater.working_dir == r"c:\foo"


class TestUpdateLayer:
    def test_update_calls_upsert_correctly(self, mocker):
        new_dataframe = pd.DataFrame(
            {
                "foo": [1, 2],
                "bar": [3, 4],
                "OBJECTID": [11, 12],
                "SHAPE": ["s1", "s2"],
            }
        )
        updater_mock = mocker.Mock(spec=load.ServiceUpdater)
        updater_mock._class_logger = logging.getLogger("mock logger")
        updater_mock.itemid = "foo123"
        updater_mock.index = 0
        updater_mock.type = "layer"
        fl_mock = mocker.Mock()
        updater_mock.service = fl_mock

        updater_mock._update_service.return_value = {"recordCount": 1}

        field_checker_mock = mocker.patch("palletjack.utils.FieldChecker")

        load.ServiceUpdater.update(updater_mock, new_dataframe, update_geometry=True)

        updater_mock._update_service.assert_called_once_with(
            updater_mock._upload_dataframe.return_value,
            new_dataframe.columns,
            upsert=True,
            upsert_matching_field="OBJECTID",
            append_fields=["foo", "bar", "OBJECTID", "SHAPE"],
            update_geometry=True,
        )

    def test_update_calls_field_checkers(self, mocker):
        new_dataframe = pd.DataFrame(
            {
                "foo": [1, 2],
                "bar": [3, 4],
                "OBJECTID": [11, 12],
            }
        )
        updater_mock = mocker.Mock(spec=load.ServiceUpdater)
        updater_mock._class_logger = logging.getLogger("mock logger")
        updater_mock.itemid = "foo123"
        updater_mock.index = 0
        updater_mock.service = mocker.Mock()
        updater_mock.service.properties.fields = {"a": [1], "b": [2]}

        updater_mock._update_service.return_value = {"recordCount": 1}

        mocker.patch.multiple(
            "palletjack.utils.FieldChecker",
            check_live_and_new_field_types_match=mocker.DEFAULT,
            check_for_non_null_fields=mocker.DEFAULT,
            check_field_length=mocker.DEFAULT,
            check_fields_present=mocker.DEFAULT,
            check_nullable_ints_shapely=mocker.DEFAULT,
        )

        load.ServiceUpdater.update(updater_mock, new_dataframe, update_geometry=True)

        load.utils.FieldChecker.check_live_and_new_field_types_match.assert_called_once_with(["foo", "bar", "OBJECTID"])
        load.utils.FieldChecker.check_for_non_null_fields.assert_called_once_with(["foo", "bar", "OBJECTID"])
        load.utils.FieldChecker.check_field_length.assert_called_once_with(["foo", "bar", "OBJECTID"])
        load.utils.FieldChecker.check_fields_present.assert_called_once_with(["foo", "bar", "OBJECTID"], add_oid=True)
        load.utils.FieldChecker.check_nullable_ints_shapely.assert_called_once()

    def test_update_no_geometry_calls_upsert_correctly(self, mocker):
        new_dataframe = pd.DataFrame(
            {
                "foo": [1, 2],
                "bar": [3, 4],
                "OBJECTID": [11, 12],
            }
        )
        updater_mock = mocker.Mock(spec=load.ServiceUpdater)
        updater_mock._class_logger = logging.getLogger("mock logger")
        updater_mock.itemid = "foo123"
        updater_mock.service = mocker.Mock()
        updater_mock.service.properties = {"geometryType": "esriGeometryPoint"}
        updater_mock.index = 0

        updater_mock._update_service.return_value = {"recordCount": 1}

        field_checker_mock = mocker.patch("palletjack.utils.FieldChecker")
        null_generator_mock = mocker.patch("palletjack.utils.get_null_geometries", return_value="Nullo")

        load.ServiceUpdater.update(updater_mock, new_dataframe, update_geometry=False)

        updater_mock._update_service.assert_called_once_with(
            updater_mock._upload_dataframe.return_value,
            new_dataframe.columns,
            upsert=True,
            upsert_matching_field="OBJECTID",
            append_fields=["foo", "bar", "OBJECTID"],
            update_geometry=False,
        )


class TestAddToLayer:
    def test_add_calls_upsert(self, mocker):
        new_dataframe = pd.DataFrame(
            {
                "foo": [1, 2],
                "bar": [3, 4],
            }
        )
        updater_mock = mocker.Mock(spec=load.ServiceUpdater)
        updater_mock._class_logger = logging.getLogger("mock logger")
        updater_mock.service_type = "layer"
        updater_mock.itemid = "foo123"
        updater_mock.service = mocker.Mock()
        updater_mock.index = 0

        updater_mock._update_service.return_value = {"recordCount": 1}

        field_checker_mock = mocker.patch("palletjack.utils.FieldChecker")

        load.ServiceUpdater.add(updater_mock, new_dataframe)

        updater_mock._update_service.assert_called_once_with(
            updater_mock._upload_dataframe.return_value,
            upsert=False,
        )

    def test_add_calls_field_checkers(self, mocker):
        new_dataframe = pd.DataFrame(
            {
                "foo": [1, 2],
                "bar": [3, 4],
            }
        )
        updater_mock = mocker.Mock(spec=load.ServiceUpdater)
        updater_mock._class_logger = logging.getLogger("mock logger")
        updater_mock.service_type = "layer"
        updater_mock.itemid = "foo123"
        updater_mock.service = mocker.Mock()
        updater_mock.service.properties.fields = {"a": [1], "b": [2]}
        updater_mock.index = 0

        updater_mock._update_service.return_value = {"recordCount": 1}

        mocker.patch.multiple(
            "palletjack.utils.FieldChecker",
            check_live_and_new_field_types_match=mocker.DEFAULT,
            check_for_non_null_fields=mocker.DEFAULT,
            check_field_length=mocker.DEFAULT,
            check_fields_present=mocker.DEFAULT,
            check_nullable_ints_shapely=mocker.DEFAULT,
        )
        load.ServiceUpdater.add(updater_mock, new_dataframe)

        load.utils.FieldChecker.check_live_and_new_field_types_match.assert_called_once_with(["foo", "bar"])
        load.utils.FieldChecker.check_for_non_null_fields.assert_called_once_with(["foo", "bar"])
        load.utils.FieldChecker.check_field_length.assert_called_once_with(["foo", "bar"])
        load.utils.FieldChecker.check_fields_present.assert_called_once_with(["foo", "bar"], add_oid=False)
        load.utils.FieldChecker.check_nullable_ints_shapely.assert_called_once()


class TestDeleteFromLayer:
    def test_remove_returns_number_of_deleted_features(self, mocker):
        updater_mock = mocker.Mock()
        updater_mock.itemid = "foo123"
        updater_mock.index = 0
        updater_mock.service = mocker.Mock()
        updater_mock.service.delete_features.return_value = {
            "deleteResults": [
                {"objectId": 11, "uniqueId": 11, "globalId": None, "success": True},
                {"objectId": 17, "uniqueId": 17, "globalId": None, "success": True},
            ]
        }  # yapf: disable

        delete_utils_mock = mocker.patch("palletjack.utils.DeleteUtils")
        delete_utils_mock.check_delete_oids_are_in_live_data.return_value = 0

        deleted = load.ServiceUpdater.remove(updater_mock, [11, 17])

        assert deleted == 2

    def test_remove_raises_on_failed_delete(self, mocker):
        updater_mock = mocker.Mock()
        updater_mock.itemid = "foo123"
        updater_mock.index = 0
        updater_mock.service.delete_features.return_value = {
            "deleteResults": [
                {"objectId": 11, "uniqueId": 11, "globalId": None, "success": True},
                {"objectId": 17, "uniqueId": 17, "globalId": None, "success": False},
            ]
        }  # yapf: disable

        mocker.patch("palletjack.utils.DeleteUtils")

        with pytest.raises(RuntimeError, match=re.escape("The following Object IDs failed to delete: [17]")):
            deleted = load.ServiceUpdater.remove(updater_mock, [11, 17])

    def test_remove_runs_proper_check_sequence(self, mocker):
        updater_mock = mocker.Mock()
        updater_mock.itemid = "foo123"
        updater_mock.index = 0
        updater_mock.service.query.return_value = {"objectIdFieldName": "OBJECTID", "objectIds": [11, 17]}
        updater_mock.service.delete_features.return_value = {
            "deleteResults": [
                {"objectId": 11, "uniqueId": 11, "globalId": None, "success": True},
            ]
        }  # yapf: disable

        deleted = load.ServiceUpdater.remove(updater_mock, [11])

        assert deleted == 1


class TestTruncateAndLoadLayer:
    def test_truncate_existing_data_normal(self, mocker):
        updater_mock = mocker.Mock()
        updater_mock.itemid = "foo123"
        updater_mock.index = 0
        updater_mock.service = mocker.Mock()

        updater_mock.service.manager.truncate.return_value = {
            "submissionTime": 123,
            "lastUpdatedTime": 124,
            "status": "Completed",
        }

        load.ServiceUpdater._truncate_existing_data(updater_mock)

    def test_truncate_existing_raises_error_on_failure(self, mocker):
        updater_mock = mocker.Mock()
        updater_mock.itemid = "foo123"
        updater_mock.service = mocker.Mock()
        updater_mock.service.manager.truncate.return_value = {
            "submissionTime": 123,
            "lastUpdatedTime": 124,
            "status": "Foo",
        }

        with pytest.raises(RuntimeError, match="Failed to truncate existing data in itemid foo123"):
            load.ServiceUpdater._truncate_existing_data(updater_mock)

    def test_truncate_existing_retries_on_HTTPError(self, mocker):
        updater_mock = mocker.Mock()
        updater_mock.itemid = "foo123"
        updater_mock.index = 0
        updater_mock.service = mocker.Mock()
        updater_mock.service.manager.truncate.side_effect = [
            urllib.error.HTTPError("url", "code", "msg", "hdrs", "fp"),
            {
                "submissionTime": 123,
                "lastUpdatedTime": 124,
                "status": "Completed",
            },
        ]

        load.ServiceUpdater._truncate_existing_data(updater_mock)

    def test_truncate_and_load_feature_service_normal(self, mocker):
        updater_mock = mocker.Mock(spec=load.ServiceUpdater)
        updater_mock._class_logger = logging.getLogger("mock logger")
        updater_mock.itemid = "foo123"
        updater_mock.index = 0
        updater_mock.service_type = "feature"
        updater_mock.service = mocker.Mock()

        new_dataframe = pd.DataFrame(columns=["Foo", "Bar"])

        mocker.patch("palletjack.utils.FieldChecker")

        updater_mock._update_service.return_value = 42

        uploaded_features = load.ServiceUpdater.truncate_and_load(updater_mock, new_dataframe)

        assert uploaded_features == 42

    def test_truncate_and_load_append_fails_save_old_true(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)

        updater_mock = mocker.Mock(spec=load.ServiceUpdater)
        updater_mock._class_logger = logging.getLogger("mock logger")
        updater_mock.service_type = "layer"
        updater_mock.itemid = "foo123"
        updater_mock.index = 0
        updater_mock.service = mocker.Mock()
        updater_mock.working_dir = Path("foo")
        updater_mock.service = mocker.Mock()
        new_dataframe = pd.DataFrame(columns=["Foo", "Bar"])

        mocker.patch("palletjack.utils.FieldChecker")
        mocker.patch("palletjack.utils.sleep")
        mocker.patch("palletjack.utils.save_to_gdb", return_value="bar_path")

        updater_mock._update_service.side_effect = RuntimeError(
            "Failed to append data. Append operation should have been rolled back."
        )

        with pytest.raises(RuntimeError, match="Failed to append data. Append operation should have been rolled back."):
            uploaded_features = load.ServiceUpdater.truncate_and_load(updater_mock, new_dataframe, save_old=True)

        updater_mock._update_service.assert_called_once_with(updater_mock._upload_dataframe.return_value, upsert=False)
        assert "Append failed. Data saved to bar_path" in caplog.text

    def test_truncate_and_load_append_fails_save_old_false(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)

        updater_mock = mocker.Mock(spec=load.ServiceUpdater)
        updater_mock._class_logger = logging.getLogger("mock logger")
        updater_mock.service_type = "layer"
        updater_mock.itemid = "foo123"
        updater_mock.index = 0
        updater_mock.service = mocker.Mock()

        new_dataframe = pd.DataFrame(columns=["Foo", "Bar"])

        mocker.patch("palletjack.utils.FieldChecker")
        mocker.patch("palletjack.utils.sleep")
        mocker.patch("palletjack.utils.save_to_gdb", return_value="bar_path")

        updater_mock._update_service.side_effect = RuntimeError(
            "Failed to append data. Append operation should have been rolled back."
        )

        with pytest.raises(RuntimeError, match="Failed to append data. Append operation should have been rolled back."):
            uploaded_features = load.ServiceUpdater.truncate_and_load(updater_mock, new_dataframe)

        updater_mock._update_service.assert_called_once_with(updater_mock._upload_dataframe.return_value, upsert=False)

        assert "Append failed. Old data not saved (save_old set to False)" in caplog.text

    def test_truncate_and_load_calls_field_checkers(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)

        updater_mock = mocker.Mock(spec=load.ServiceUpdater)
        updater_mock._class_logger = logging.getLogger("mock logger")
        updater_mock.itemid = "foo123"
        updater_mock.service_type = "layer"
        updater_mock.service = mocker.Mock()
        updater_mock.service.properties.fields = {"a": [1], "b": [2]}
        updater_mock.index = 0

        new_dataframe = pd.DataFrame(
            {
                "Foo": [1, 2],
                "Bar": [3, 4],
            }
        )

        mocker.patch("palletjack.utils.sleep")
        mocker.patch.multiple(
            "palletjack.utils.FieldChecker",
            check_live_and_new_field_types_match=mocker.DEFAULT,
            check_for_non_null_fields=mocker.DEFAULT,
            check_field_length=mocker.DEFAULT,
            check_fields_present=mocker.DEFAULT,
            check_nullable_ints_shapely=mocker.DEFAULT,
        )

        updater_mock._update_service.return_value = 42

        uploaded_features = load.ServiceUpdater.truncate_and_load(updater_mock, new_dataframe)

        load.utils.FieldChecker.check_live_and_new_field_types_match.assert_called_once_with(["Foo", "Bar"])
        load.utils.FieldChecker.check_for_non_null_fields.assert_called_once_with(["Foo", "Bar"])
        load.utils.FieldChecker.check_field_length.assert_called_once_with(["Foo", "Bar"])
        load.utils.FieldChecker.check_fields_present.assert_called_once_with(["Foo", "Bar"], add_oid=False)
        load.utils.FieldChecker.check_nullable_ints_shapely.assert_called_once()

        assert uploaded_features == 42

    def test_truncate_and_load_doesnt_raise_on_empty_column(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)

        new_dataframe = pd.DataFrame(
            {"id": [0, 1, 2], "floats": [np.nan, np.nan, np.nan], "x": [21, 22, 23], "y": [31, 32, 33]}
        )
        spatial_df = pd.DataFrame.spatial.from_xy(new_dataframe, "x", "y")

        updater_mock = mocker.Mock(spec=load.ServiceUpdater)
        updater_mock._class_logger = logging.getLogger("mock logger")
        updater_mock.itemid = "foo123"
        updater_mock.index = 0
        updater_mock.service_type = "layer"
        updater_mock.service = mocker.Mock()

        updater_mock.service.properties.fields = [
            {
                "name": "id",
                "type": "esriFieldTypeInteger",
                "nullable": True,
                "defaultValue": None,
            },
            {
                "name": "floats",
                "type": "esriFieldTypeDouble",
                "nullable": True,
                "defaultValue": None,
            },
            {
                "name": "x",
                "type": "esriFieldTypeInteger",
                "nullable": True,
                "defaultValue": None,
            },
            {
                "name": "y",
                "type": "esriFieldTypeInteger",
                "nullable": True,
                "defaultValue": None,
            },
        ]
        updater_mock.service.properties.geometryType = "esriGeometryPoint"
        updater_mock.service.properties.extent.spatialReference.latestWkid = 4326

        mocker.patch("palletjack.utils.sleep")

        updater_mock._update_service.return_value = {"recordCount": 42}

        #: Should not raise
        uploaded_features = load.ServiceUpdater.truncate_and_load(updater_mock, spatial_df)


class TestAttachments:
    def test_create_attachment_action_df_adds_for_blank_existing_name(self, mocker):
        input_df = pd.DataFrame(
            {
                "NAME": [np.nan],
                "new_path": ["bee/foo.png"],
            }
        )

        ops_df = load.FeatureServiceAttachmentsUpdater._create_attachment_action_df(mocker.Mock(), input_df, "new_path")

        test_df = pd.DataFrame(
            {
                "NAME": [np.nan],
                "new_path": ["bee/foo.png"],
                "new_filename": ["foo.png"],
                "operation": ["add"],
            }
        )

        tm.assert_frame_equal(ops_df, test_df)

    def test_create_attachment_action_df_overwrites_for_different_existing_name(self, mocker):
        input_df = pd.DataFrame(
            {
                "NAME": ["bar.png"],
                "new_path": ["bee/foo.png"],
            }
        )

        ops_df = load.FeatureServiceAttachmentsUpdater._create_attachment_action_df(mocker.Mock(), input_df, "new_path")

        test_df = pd.DataFrame(
            {
                "NAME": ["bar.png"],
                "new_path": ["bee/foo.png"],
                "new_filename": ["foo.png"],
                "operation": ["overwrite"],
            }
        )

        tm.assert_frame_equal(ops_df, test_df)

    def test_create_attachment_action_df_does_nothing_for_same_name(self, mocker):
        input_df = pd.DataFrame(
            {
                "NAME": ["foo.png"],
                "new_path": ["bee/foo.png"],
            }
        )

        ops_df = load.FeatureServiceAttachmentsUpdater._create_attachment_action_df(mocker.Mock(), input_df, "new_path")

        test_df = pd.DataFrame(
            {
                "NAME": ["foo.png"],
                "new_path": ["bee/foo.png"],
                "new_filename": ["foo.png"],
                "operation": [np.nan],
            }
        )
        test_df["operation"] = test_df["operation"].astype(object)

        tm.assert_frame_equal(ops_df, test_df)

    def test_create_attachment_action_df_does_all_three_ops(self, mocker):
        input_df = pd.DataFrame(
            {
                "NAME": ["foo.png", "bar.png", np.nan],
                "new_path": ["bee/foo.png", "bee/baz.png", "bee/bin.png"],
            }
        )

        ops_df = load.FeatureServiceAttachmentsUpdater._create_attachment_action_df(mocker.Mock(), input_df, "new_path")

        test_df = pd.DataFrame(
            {
                "NAME": ["foo.png", "bar.png", np.nan],
                "new_path": ["bee/foo.png", "bee/baz.png", "bee/bin.png"],
                "new_filename": ["foo.png", "baz.png", "bin.png"],
                "operation": [np.nan, "overwrite", "add"],
            }
        )

        tm.assert_frame_equal(ops_df, test_df)

    def test_create_attachment_action_df_do_nothing_after_others(self, mocker):
        input_df = pd.DataFrame(
            {
                "NAME": ["bar.png", np.nan, "foo.png"],
                "new_path": ["bee/baz.png", "bee/bin.png", "bee/foo.png"],
            }
        )

        ops_df = load.FeatureServiceAttachmentsUpdater._create_attachment_action_df(mocker.Mock(), input_df, "new_path")

        test_df = pd.DataFrame(
            {
                "NAME": ["bar.png", np.nan, "foo.png"],
                "new_path": ["bee/baz.png", "bee/bin.png", "bee/foo.png"],
                "new_filename": ["baz.png", "bin.png", "foo.png"],
                "operation": ["overwrite", "add", np.nan],
            }
        )

        tm.assert_frame_equal(ops_df, test_df)

    def test_get_live_data_from_join_field_values_only_gets_matching_data(self, mocker):
        live_features_df = pd.DataFrame(
            {
                "OBJECTID": [1, 2, 3],
                "GlobalID": ["guid1", "guid2", "guid3"],
                "attachment_key": [11, 12, 13],
                "deleted": ["a", "b", "c"],
            }
        )

        attachments_df = pd.DataFrame(
            {
                "attachment_key": [12, 13],
                "attachments": ["foo", "bar"],
            }
        )

        live_data_subset = load.FeatureServiceAttachmentsUpdater._get_live_oid_and_guid_from_join_field_values(
            mocker.Mock(), live_features_df, "attachment_key", attachments_df
        )

        test_df = pd.DataFrame(
            {
                "OBJECTID": [2, 3],
                "GlobalID": ["guid2", "guid3"],
                "attachment_key": [12, 13],
                "attachments": ["foo", "bar"],
            }
        )

        tm.assert_frame_equal(live_data_subset, test_df)

    def test_get_current_attachment_info_by_oid_includes_nans_for_features_wo_attachments(self, mocker):
        live_attachments = [
            {
                "PARENTOBJECTID": 1,
                "PARENTGLOBALID": "parentguid1",
                "ID": 111,
                "NAME": "foo.png",
                "CONTENTTYPE": "image/png",
                "SIZE": 42,
                "KEYWORDS": "",
                "IMAGE_PREVIEW": "preview1",
                "GLOBALID": "guid1",
                "DOWNLOAD_URL": "url1",
            },
            {
                "PARENTOBJECTID": 2,
                "PARENTGLOBALID": "parentguid2",
                "ID": 222,
                "NAME": "bar.png",
                "CONTENTTYPE": "image/png",
                "SIZE": 42,
                "KEYWORDS": "",
                "IMAGE_PREVIEW": "preview2",
                "GLOBALID": "guid2",
                "DOWNLOAD_URL": "url2",
            },
        ]

        updater_mock = mocker.Mock()
        updater_mock.feature_layer.attachments.search.return_value = live_attachments

        live_data_subset_df = pd.DataFrame(
            {
                "OBJECTID": [1, 2, 3],
                "GlobalID": ["guid1", "guid2", "guid3"],
                "attachment_key": [11, 12, 13],
                "attachments": ["fee", "ber", "boo"],
            }
        )

        current_attachments_df = load.FeatureServiceAttachmentsUpdater._get_current_attachment_info_by_oid(
            updater_mock, live_data_subset_df
        )

        test_df = pd.DataFrame(
            {
                "OBJECTID": [1, 2, 3],
                "GlobalID": ["guid1", "guid2", "guid3"],
                "attachment_key": [11, 12, 13],
                "attachments": ["fee", "ber", "boo"],
                "PARENTOBJECTID": [1.0, 2.0, np.nan],
                "NAME": ["foo.png", "bar.png", np.nan],
                "ID": [111, 222, pd.NA],
            }
        )
        test_df["ID"] = test_df["ID"].astype("Int64")

        tm.assert_frame_equal(current_attachments_df, test_df)

    def test_check_attachment_dataframe_for_invalid_column_names_doesnt_raise_with_valid_names(self, mocker):
        dataframe = pd.DataFrame(columns=["foo", "bar"])
        invalid_names = ["baz", "boo"]
        load.FeatureServiceAttachmentsUpdater._check_attachment_dataframe_for_invalid_column_names(
            dataframe, invalid_names
        )

    def test_check_attachment_dataframe_for_invalid_column_names_raises_with_one_invalid(self, mocker):
        dataframe = pd.DataFrame(columns=["foo", "bar"])
        invalid_names = ["foo", "boo"]
        with pytest.raises(RuntimeError) as exc_info:
            load.FeatureServiceAttachmentsUpdater._check_attachment_dataframe_for_invalid_column_names(
                dataframe, invalid_names
            )
        assert exc_info.value.args[0] == "Attachment dataframe contains the following invalid names: ['foo']"

    def test_check_attachment_dataframe_for_invalid_column_names_raises_with_all_invalid(self, mocker):
        dataframe = pd.DataFrame(columns=["foo", "bar"])
        invalid_names = ["foo", "bar"]
        with pytest.raises(RuntimeError) as exc_info:
            load.FeatureServiceAttachmentsUpdater._check_attachment_dataframe_for_invalid_column_names(
                dataframe, invalid_names
            )
        assert exc_info.value.args[0] == "Attachment dataframe contains the following invalid names: ['foo', 'bar']"

    def test_add_attachments_by_oid_adds_and_doesnt_warn(self, mocker):
        action_df = pd.DataFrame(
            {
                "OBJECTID": [1, 2],
                "operation": ["add", "add"],
                "path": ["path1", "path2"],
            }
        )

        result_dict = [
            {"addAttachmentResult": {"success": True}},
            {"addAttachmentResult": {"success": True}},
        ]

        updater_mock = mocker.Mock()
        updater_mock.feature_layer.attachments.add.side_effect = result_dict

        with warnings.catch_warnings() as warning:
            count = load.FeatureServiceAttachmentsUpdater._add_attachments_by_oid(updater_mock, action_df, "path")

        assert count == 2
        assert not warning

    def test_add_attachments_by_oid_warns_on_failure_and_doesnt_count_that_one_and_continues(self, mocker):
        action_df = pd.DataFrame(
            {
                "OBJECTID": [1, 2, 3],
                "operation": ["add", "add", "add"],
                "path": ["path1", "path2", "path3"],
            }
        )

        result_dict = [
            {"addAttachmentResult": {"success": True}},
            {"addAttachmentResult": {"success": False}},
            {"addAttachmentResult": {"success": True}},
        ]

        feature_layer_mock = mocker.Mock()
        feature_layer_mock.attachments.add.side_effect = result_dict

        updater = load.FeatureServiceAttachmentsUpdater(mocker.Mock())
        updater.feature_layer = feature_layer_mock

        with pytest.warns(UserWarning, match="Failed to attach path2 to OID 2"):
            count = updater._add_attachments_by_oid(action_df, "path")

        assert count == 2
        assert updater.failed_dict == {2: ("add", "path2")}

    def test_add_attachments_by_oid_handles_internal_agol_errors(self, mocker, caplog):
        action_df = pd.DataFrame(
            {
                "OBJECTID": [1, 2],
                "operation": ["add", "add"],
                "path": ["path1", "path2"],
            }
        )

        feature_layer_mock = mocker.Mock()
        feature_layer_mock.attachments.add.side_effect = [
            RuntimeError("foo"),
            {"addAttachmentResult": {"success": True}},
        ]

        updater = load.FeatureServiceAttachmentsUpdater(mocker.Mock())
        updater.feature_layer = feature_layer_mock

        count = updater._add_attachments_by_oid(action_df, "path")
        assert count == 1
        assert "AGOL error while adding path1 to OID 1" in caplog.text
        assert "foo" in caplog.text
        assert updater.failed_dict == {1: ("add", "path1")}

    def test_add_attachments_by_oid_skips_overwrite_and_nan(self, mocker):
        action_df = pd.DataFrame(
            {
                "OBJECTID": [1, 2, 3],
                "operation": ["add", "overwrite", np.nan],
                "path": ["path1", "path2", "path3"],
            }
        )

        result_dict = [
            {"addAttachmentResult": {"success": True}},
        ]

        updater_mock = mocker.Mock()
        updater_mock.feature_layer.attachments.add.side_effect = result_dict

        count = load.FeatureServiceAttachmentsUpdater._add_attachments_by_oid(updater_mock, action_df, "path")

        assert updater_mock.feature_layer.attachments.add.call_count == 1
        assert count == 1

    def test_overwrite_attachments_by_oid_overwrites_and_doesnt_warn(self, mocker):
        action_df = pd.DataFrame(
            {
                "OBJECTID": [1, 2],
                "operation": ["overwrite", "overwrite"],
                "path": ["path1", "path2"],
                "ID": ["existing1", "existing2"],
                "NAME": ["oldname1", "oldname2"],
            }
        )

        result_dict = [
            {"updateAttachmentResult": {"success": True}},
            {"updateAttachmentResult": {"success": True}},
        ]

        updater_mock = mocker.Mock()
        updater_mock.feature_layer.attachments.update.side_effect = result_dict

        with warnings.catch_warnings() as warning:
            count = load.FeatureServiceAttachmentsUpdater._overwrite_attachments_by_oid(updater_mock, action_df, "path")

        assert count == 2
        assert not warning

    def test_overwrite_attachments_by_oid_warns_on_failure_and_doesnt_count_that_one_and_continues(self, mocker):
        action_df = pd.DataFrame(
            {
                "OBJECTID": [1, 2, 3],
                "operation": ["overwrite", "overwrite", "overwrite"],
                "path": ["path1", "path2", "path3"],
                "ID": [11, 22, 33],
                "NAME": ["oldname1", "oldname2", "oldname3"],
            }
        )

        result_dict = [
            {"updateAttachmentResult": {"success": True}},
            {"updateAttachmentResult": {"success": False}},
            {"updateAttachmentResult": {"success": True}},
        ]

        feature_layer_mock = mocker.Mock()
        feature_layer_mock.attachments.update.side_effect = result_dict

        updater = load.FeatureServiceAttachmentsUpdater(mocker.Mock())
        updater.feature_layer = feature_layer_mock

        with pytest.warns(UserWarning, match="Failed to update oldname2, attachment ID 22, on OID 2 with path2"):
            count = updater._overwrite_attachments_by_oid(action_df, "path")

        assert count == 2
        assert updater.failed_dict == {2: ("update", "path2")}

    def test_overwrite_attachments_by_oid_handles_internal_agol_errors(self, mocker, caplog):
        action_df = pd.DataFrame(
            {
                "OBJECTID": [1, 2],
                "operation": ["overwrite", "overwrite"],
                "path": ["path1", "path2"],
                "ID": [11, 22],
                "NAME": ["oldname1", "oldname2"],
            }
        )

        feature_layer_mock = mocker.Mock()
        feature_layer_mock.attachments.update.side_effect = [
            RuntimeError("foo"),
            {"updateAttachmentResult": {"success": True}},
        ]

        updater = load.FeatureServiceAttachmentsUpdater(mocker.Mock())
        updater.feature_layer = feature_layer_mock

        count = updater._overwrite_attachments_by_oid(action_df, "path")
        assert count == 1
        assert "AGOL error while overwriting oldname1 (attachment ID 11) on OID 1 with path1" in caplog.text
        assert "foo" in caplog.text
        assert updater.failed_dict == {1: ("update", "path1")}

    def test_overwrite_attachments_by_oid_skips_add_and_nan(self, mocker):
        action_df = pd.DataFrame(
            {
                "OBJECTID": [1, 2, 3],
                "operation": ["add", "overwrite", np.nan],
                "path": ["path1", "path2", "path3"],
                "ID": [np.nan, "existing2", "existing3"],
                "NAME": ["oldname1", "oldname2", "oldname3"],
            }
        )

        result_dict = [
            {"updateAttachmentResult": {"success": True}},
        ]

        updater_mock = mocker.Mock()
        updater_mock.feature_layer.attachments.update.side_effect = result_dict

        count = load.FeatureServiceAttachmentsUpdater._overwrite_attachments_by_oid(updater_mock, action_df, "path")

        assert updater_mock.feature_layer.attachments.update.call_count == 1
        assert count == 1

    def test_create_attachments_dataframe_subsets_and_crafts_paths_properly(self, mocker):
        input_df = pd.DataFrame(
            {
                "join": [1, 2, 3],
                "pic": ["foo.png", "bar.png", "baz.png"],
                "data": [11.0, 12.0, 13.0],
            }
        )

        attachment_df = load.FeatureServiceAttachmentsUpdater.build_attachments_dataframe(
            input_df, "join", "pic", "/foo/bar"
        )

        test_df = pd.DataFrame(
            {
                "join": [1, 2, 3],
                "pic": ["foo.png", "bar.png", "baz.png"],
                "full_file_path": [Path("/foo/bar/foo.png"), Path("/foo/bar/bar.png"), Path("/foo/bar/baz.png")],
            }
        )

        #: Column of path objects won't test equal in assert_frame_equal, so we make lists of their str representations
        #: and compare those separately from the rest of the dataframe
        other_fields = ["join", "pic"]
        tm.assert_frame_equal(attachment_df[other_fields], test_df[other_fields])
        assert [str(path) for path in attachment_df["full_file_path"]] == [
            str(path) for path in test_df["full_file_path"]
        ]

    def test_create_attachments_dataframe_drops_missing_attachments(self, mocker):
        input_df = pd.DataFrame(
            {
                "join": [1, 2, 3],
                "pic": ["foo.png", None, ""],
            }
        )

        attachment_df = load.FeatureServiceAttachmentsUpdater.build_attachments_dataframe(
            input_df, "join", "pic", "/foo/bar"
        )

        test_df = pd.DataFrame(
            {
                "join": [1],
                "pic": ["foo.png"],
                "full_file_path": [Path("/foo/bar/foo.png")],
            }
        )

        #: Column of path objects won't test equal in assert_frame_equal, so we make lists of their str representations
        #: and compare those separately from the rest of the dataframe
        other_fields = ["join", "pic"]
        tm.assert_frame_equal(attachment_df[other_fields], test_df[other_fields])
        assert [str(path) for path in attachment_df["full_file_path"]] == [
            str(path) for path in test_df["full_file_path"]
        ]


class TestUpdateService:
    def test_update_service_calls_append_with_proper_args(self, mocker):
        expected_kwargs = {
            "item_id": "1234",
            "upload_format": "filegdb",
            "source_table_name": "upload",
            "upsert": True,
            "rollback": True,
            "return_messages": True,
        }
        updater_mock = mocker.Mock()
        updater_mock.service = mocker.Mock()
        updater_mock.service.append.return_value = (True, {"recordCount": 42})

        gdb_item_mock = mocker.Mock()
        gdb_item_mock.id = "1234"

        load.ServiceUpdater._update_service(updater_mock, gdb_item_mock, upsert=True)

        updater_mock.service.append.assert_called_once_with(**expected_kwargs)

    def test_update_service_retries_on_exception(self, mocker):
        updater_mock = mocker.Mock()
        updater_mock.service = mocker.Mock()
        updater_mock.service.append.side_effect = [Exception, (True, {"recordCount": 42})]
        mocker.patch("palletjack.utils.sleep")

        load.ServiceUpdater._update_service(updater_mock, mocker.Mock(), upsert=True)

        assert updater_mock.service.append.call_count == 2

    def test_update_service_raises_on_False_result(self, mocker):
        expected_inner_error = "Append failed but did not error"
        expected_outer_error = "Failed to append data from gdb, changes should have been rolled back"
        updater_mock = mocker.Mock()
        updater_mock.service = mocker.Mock()
        updater_mock.service.append.return_value = (False, {"message": "foo"})

        gdb_item_mock = mocker.Mock()
        gdb_item_mock.id = "1234"

        with pytest.raises(RuntimeError) as exc_info:
            load.ServiceUpdater._update_service(updater_mock, gdb_item_mock, upsert=True)

        assert exc_info.value.args[0] == expected_outer_error
        assert exc_info.value.__cause__.args[0] == expected_inner_error

    def test_update_service_raises_on_upsert_field_not_in_append_fields(self, mocker):
        append_kwargs = {"upsert_matching_field": "foo", "append_fields": ["bar", "baz"], "upsert": True}
        with pytest.raises(ValueError) as exc_info:
            load.ServiceUpdater._update_service(mocker.Mock(), mocker.Mock(), **append_kwargs)

        assert (
            exc_info.value.args[0] == "Upsert matching field foo not found in either append fields or existing fields."
        )

    def test_update_service_raises_on_upsert_field_not_in_dataframe_columns(self, mocker):
        append_kwargs = {"upsert_matching_field": "foo", "append_fields": ["foo", "bar"], "upsert": True}
        df = pd.DataFrame(columns=["bar", "baz"])
        with pytest.raises(ValueError) as exc_info:
            load.ServiceUpdater._update_service(mocker.Mock(), mocker.Mock(), df.columns, **append_kwargs)

        assert (
            exc_info.value.args[0] == "Upsert matching field foo not found in either append fields or existing fields."
        )

    def test_update_service_raises_on_upsert_field_not_in_dataframe_columns_and_append_fields(self, mocker):
        append_kwargs = {"upsert_matching_field": "foo", "append_fields": ["bar", "baz"], "upsert": True}
        df = pd.DataFrame(columns=["bar", "baz"])
        with pytest.raises(ValueError) as exc_info:
            load.ServiceUpdater._update_service(mocker.Mock(), mocker.Mock(), df.columns, **append_kwargs)

        assert (
            exc_info.value.args[0] == "Upsert matching field foo not found in either append fields or existing fields."
        )


class TestUploadDataframe:
    def test_upload_dataframe_calls_cleanup_with_zipped_path(self, mocker):
        updater_mock = mocker.Mock()

        load.ServiceUpdater._upload_dataframe(updater_mock, mocker.Mock())

        assert updater_mock._cleanup.called_once_with(updater_mock._upload_gdb.return_value)


class TestCleanup:
    def test_cleanup_calls_item_delete(self, mocker):
        gdb_item_mock = mocker.Mock()
        updater_mock = mocker.Mock()

        load.ServiceUpdater._cleanup(updater_mock, gdb_item=gdb_item_mock)

        gdb_item_mock.delete.assert_called_once()

    def test_cleanup_removes_zipped_file_on_filesystem(self, mocker):
        gdb_path_mock = mocker.Mock(spec=Path)

        updater_mock = mocker.Mock()

        load.ServiceUpdater._cleanup(updater_mock, zipped_gdb_path=gdb_path_mock)

        gdb_path_mock.unlink.assert_called_once()

    def test_cleanup_does_both(self, mocker):
        gdb_item_mock = mocker.Mock()
        gdb_path_mock = mocker.Mock(spec=Path)

        updater_mock = mocker.Mock()

        load.ServiceUpdater._cleanup(updater_mock, gdb_item=gdb_item_mock, zipped_gdb_path=gdb_path_mock)

        gdb_item_mock.delete.assert_called_once()
        gdb_path_mock.unlink.assert_called_once()

    def test_cleanup_warns_on_exceptions(self, mocker):
        gdb_item_mock = mocker.Mock()
        gdb_item_mock.delete.side_effect = Exception("foo")
        gdb_path_mock = mocker.Mock(spec=Path)
        gdb_path_mock.unlink.side_effect = Exception("bar")

        updater_mock = mocker.Mock()
        updater_mock._class_logger = logging.getLogger("mock logger")

        with warnings.catch_warnings(record=True) as warning_list:
            load.ServiceUpdater._cleanup(updater_mock, gdb_item=gdb_item_mock, zipped_gdb_path=gdb_path_mock)

        assert len(warning_list) == 4
        assert "Error deleting gdb item" in str(warning_list[0].message)
        assert "foo" in str(warning_list[1].message)
        assert "Error deleting zipped gdb" in str(warning_list[2].message)
        assert "bar" in str(warning_list[3].message)


class TestColorRampReclassifier:
    def test_get_layer_id_returns_match_single_layer(self, mocker):
        layers = {
            "operationalLayers": [
                {"title": "foo"},
            ],
        }
        get_data_mock = mocker.Mock(return_value=layers)
        webmap_item_mock = mocker.Mock()
        webmap_item_mock.get_data = get_data_mock
        reclassifier = load.ColorRampReclassifier(webmap_item_mock, "gis")

        layer_id = reclassifier._get_layer_id("foo")
        assert layer_id == 0

    def test_get_layer_id_returns_match_many_layers(self, mocker):
        layers = {
            "operationalLayers": [
                {"title": "foo"},
                {"title": "bar"},
                {"title": "baz"},
            ],
        }
        get_data_mock = mocker.Mock(return_value=layers)
        webmap_item_mock = mocker.Mock()
        webmap_item_mock.get_data = get_data_mock
        reclassifier = load.ColorRampReclassifier(webmap_item_mock, "gis")

        layer_id = reclassifier._get_layer_id("bar")
        assert layer_id == 1

    def test_get_layer_id_returns_first_match(self, mocker):
        layers = {
            "operationalLayers": [
                {"title": "foo"},
                {"title": "bar"},
                {"title": "bar"},
            ],
        }
        get_data_mock = mocker.Mock(return_value=layers)
        webmap_item_mock = mocker.Mock()
        webmap_item_mock.get_data = get_data_mock
        reclassifier = load.ColorRampReclassifier(webmap_item_mock, "gis")

        layer_id = reclassifier._get_layer_id("bar")
        assert layer_id == 1

    def test_get_layer_id_raises_error_when_not_found(self, mocker):
        layers = {
            "operationalLayers": [
                {"title": "bar"},
            ],
        }
        get_data_mock = mocker.Mock(return_value=layers)
        webmap_item_mock = mocker.Mock()
        webmap_item_mock.title = "test map"
        webmap_item_mock.get_data = get_data_mock
        reclassifier = load.ColorRampReclassifier(webmap_item_mock, "gis")

        with pytest.raises(ValueError) as error_info:
            layer_id = reclassifier._get_layer_id("foo")

        assert 'Could not find "foo" in test map' in str(error_info.value)

    def test_calculate_new_stops_with_manual_numbers(self):
        dataframe = pd.DataFrame({"numbers": [100, 300, 500, 700, 900]})

        stops = load.ColorRampReclassifier._calculate_new_stops(dataframe, "numbers", 5)

        assert stops == [100, 279, 458, 637, 816]

    def test_calculate_new_stops_mismatched_column_raises_error(self):
        dataframe = pd.DataFrame({"numbers": [100, 300, 500, 700, 900]})

        with pytest.raises(ValueError) as error_info:
            stops = load.ColorRampReclassifier._calculate_new_stops(dataframe, "foo", 5)
            assert "Column `foo` not in dataframe`" in str(error_info)

    def test_update_stops_values(self, mocker):
        # renderer = data['operationalLayers'][layer_number]['layerDefinition']['drawingInfo']['renderer']
        # stops = renderer['visualVariables'][0]['stops']

        data = {
            "operationalLayers": [
                {
                    "layerDefinition": {
                        "drawingInfo": {
                            "renderer": {
                                "visualVariables": [{"stops": [{"value": 0}, {"value": 1}, {"value": 2}, {"value": 3}]}]
                            }
                        }
                    }
                }
            ],
        }
        get_data_mock = mocker.Mock(return_value=data)
        webmap_item_mock = mocker.Mock()
        webmap_item_mock.get_data = get_data_mock
        update_mock = mocker.Mock()
        webmap_item_mock.update = update_mock
        reclassifier = load.ColorRampReclassifier(webmap_item_mock, "gis")

        reclassifier._update_stop_values(0, [100, 200, 300, 400])

        data["operationalLayers"][0]["layerDefinition"]["drawingInfo"]["renderer"]["visualVariables"][0]["stops"] = [
            {"value": 100},
            {"value": 200},
            {"value": 300},
            {"value": 400},
        ]

        assert update_mock.called_with(item_properties={"text": json.dumps(data)})


class TestGDBStuff:
    def test__save_to_gdb_and_zip_uses_correct_directories(self, mocker):
        expected_call_args = [Path("/foo/bar/upload.gdb"), "zip"]
        expected_call_kwargs = {"root_dir": Path("/foo/bar"), "base_dir": "upload.gdb"}

        updater_mock = mocker.Mock()
        updater_mock.working_dir = "/foo/bar"
        updater_mock.service = mocker.Mock()

        mocker.patch("palletjack.utils.convert_to_gdf")
        shutil_mock = mocker.patch("palletjack.load.shutil")

        foo = load.ServiceUpdater._save_to_gdb_and_zip(updater_mock, mocker.Mock())

        shutil_mock.make_archive.assert_called_once_with(*expected_call_args, **expected_call_kwargs)

    def test__save_to_gdb_and_zip_raises_on_gdf_write_error(self, mocker):
        gdb_path = Path("/foo/bar/upload.gdb")
        expected_error = f"Error writing layer to {gdb_path}. Verify /foo/bar exists and is writable."

        updater_mock = mocker.Mock()
        updater_mock.working_dir = "/foo/bar"
        updater_mock.service = mocker.Mock()

        gdf_mock = mocker.patch("palletjack.utils.convert_to_gdf").return_value
        gdf_mock.to_file.side_effect = pyogrio.errors.DataSourceError

        with pytest.raises(ValueError, match=re.escape(expected_error)):
            load.ServiceUpdater._save_to_gdb_and_zip(updater_mock, mocker.Mock())

    def test__save_to_gdb_and_zip_raises_on_zip_error(self, mocker):
        updater_mock = mocker.Mock()
        updater_mock.working_dir = "/foo/bar"
        updater_mock.service = mocker.Mock()
        mocker.patch("palletjack.utils.convert_to_gdf")
        mocker.patch("palletjack.load.shutil.make_archive", side_effect=OSError("io error"))

        gdb_path = Path("/foo/bar/upload.gdb")
        with pytest.raises(ValueError, match=re.escape(f"Error zipping {gdb_path}")) as exc_info:
            foo = load.ServiceUpdater._save_to_gdb_and_zip(updater_mock, mocker.Mock())

        assert exc_info.value.__cause__.args[0] == "io error"

    def test__save_to_gdb_and_zip_raises_missing_working_dir_attribute(self, mocker):
        expected_outer_error = "working_dir not specified"
        expected_inner_error = "expected str, bytes or os.PathLike object, not NoneType"

        updater_mock = mocker.Mock()
        updater_mock.working_dir = None
        updater_mock.service = mocker.Mock()

        with pytest.raises(AttributeError, match=re.escape(expected_outer_error)) as exc_info:
            load.ServiceUpdater._save_to_gdb_and_zip(updater_mock, mocker.Mock())

        assert exc_info.value.__cause__.args[0] == expected_inner_error

    def test__save_to_gdb_and_zip_promotes_to_polyline(self, mocker):
        updater_mock = mocker.Mock()
        updater_mock.working_dir = "/foo/bar"
        mocker.patch("palletjack.load.shutil.make_archive")
        gdf_mock = mocker.patch("palletjack.utils.convert_to_gdf").return_value
        gdf_mock.geometry.geom_type = pd.Series(["LineString", "MultiLineString"])

        gdb_path = Path("/foo/bar/upload.gdb")

        load.ServiceUpdater._save_to_gdb_and_zip(updater_mock, mocker.Mock())

        assert gdf_mock.to_file.called_with(
            gdb_path,
            layer="upload",
            engine="pyogrio",
            driver="FileGDB",
            promote_to_multi=True,
        )

    def test__save_to_gdb_and_zip_doesnt_promote_points(self, mocker):
        updater_mock = mocker.Mock()
        updater_mock.working_dir = "/foo/bar"
        mocker.patch("palletjack.load.shutil.make_archive")
        gdf_mock = mocker.patch("palletjack.utils.convert_to_gdf").return_value
        gdf_mock.geometry.geom_type = pd.Series(["point", "point"])

        gdb_path = Path("/foo/bar/upload.gdb")

        load.ServiceUpdater._save_to_gdb_and_zip(updater_mock, mocker.Mock())

        assert gdf_mock.to_file.called_with(
            gdb_path,
            layer="upload",
            engine="pyogrio",
            driver="FileGDB",
            promote_to_multi=False,
        )

    def test__save_to_gdb_and_zip_doesnt_promote_multipoints(self, mocker):
        updater_mock = mocker.Mock()
        updater_mock.working_dir = "/foo/bar"
        mocker.patch("palletjack.load.shutil.make_archive")
        gdf_mock = mocker.patch("palletjack.utils.convert_to_gdf").return_value
        gdf_mock.geometry.geom_type = pd.Series(["mutlipoint", "mutlipoint"])

        gdb_path = Path("/foo/bar/upload.gdb")

        load.ServiceUpdater._save_to_gdb_and_zip(updater_mock, mocker.Mock())

        assert gdf_mock.to_file.called_with(
            gdb_path,
            layer="upload",
            engine="pyogrio",
            driver="FileGDB",
            promote_to_multi=False,
        )

    def test__upload_gdb_calls_add_default_name(self, mocker):
        gdb_path = Path("/foo/bar/upload.gdb")
        expected_call_kwargs = {
            "item_properties": {
                "type": "File Geodatabase",
                "title": "palletjack Temporary gdb upload",
                "snippet": "Temporary gdb upload from palletjack",
            },
            "data": gdb_path,
        }

        mocker.patch("palletjack.load.arcgis")
        gis_mock = mocker.Mock()
        gis_mock.content.search.return_value = []
        updater = load.ServiceUpdater(gis_mock, "abc", service_type="table")

        foo = updater._upload_gdb(gdb_path)

        gis_mock.content.add.assert_called_once_with(**expected_call_kwargs)

    def test__upload_gdb_calls_add_custom_name(self, mocker):
        gdb_path = Path("/foo/bar/upload.gdb")
        expected_call_kwargs = {
            "item_properties": {
                "type": "File Geodatabase",
                "title": "foo Temporary gdb upload",
                "snippet": "Temporary gdb upload from palletjack",
            },
            "data": gdb_path,
        }

        mocker.patch("palletjack.load.arcgis")
        gis_mock = mocker.Mock()
        gis_mock.content.search.return_value = []
        updater = load.ServiceUpdater(gis_mock, "abc", gdb_item_prefix="foo")

        foo = updater._upload_gdb(gdb_path)

        gis_mock.content.add.assert_called_once_with(**expected_call_kwargs)

    def test__upload_gdb_raises_on_agol_error(self, mocker):
        gdb_path = Path("/foo/bar/upload.gdb")
        updater_mock = mocker.Mock()
        updater_mock.gis.content.search.return_value = []
        updater_mock.gis.content.add.side_effect = [Exception("foo")] * 4
        mocker.patch("palletjack.utils.sleep")

        with pytest.raises(RuntimeError) as exc_info:
            load.ServiceUpdater._upload_gdb(updater_mock, gdb_path)

        assert exc_info.value.args[0] == f"Error uploading {gdb_path} to AGOL"
        assert updater_mock.gis.content.add.call_count == 4  #: retries

    def test__upload_gdb_deletes_existing_item(self, mocker):
        gdb_path = Path("/foo/bar/upload.gdb")
        updater_mock = mocker.Mock()
        deleteFunction = mocker.Mock()
        updater_mock.gis.content.search.return_value = [mocker.Mock(id="1234", delete=deleteFunction)]
        updater_mock.gis.content.add.return_value = mocker.Mock()
        updater_mock.gdb_item_prefix = "palletjack"
        updater_mock.gis.users.me.username = "test_user"

        load.ServiceUpdater._upload_gdb(updater_mock, gdb_path)

        deleteFunction.assert_called_once()
        updater_mock.gis.content.add.assert_called_once()

    def test__cleanup_deletes_agol_and_file(self, mocker):
        zipped_path = Path("/foo/bar/upload.gdb.zip")
        gdb_item_mock = mocker.Mock()

        load.ServiceUpdater._cleanup(mocker.Mock(), gdb_item_mock, zipped_path)

        gdb_item_mock.delete.assert_called_once()

    def test__cleanup_warns_on_agol_error_and_continues(self, mocker):
        expected_warning = "Error deleting gdb item 1234 from AGOL"
        gdb_item_mock = mocker.Mock(id="1234")
        gdb_item_mock.delete.side_effect = [RuntimeError("Unable to delete item.")]
        zipped_path = Path("/foo/bar/upload.gdb.zip")

        with pytest.warns(UserWarning, match=re.escape(expected_warning)):
            load.ServiceUpdater._cleanup(mocker.Mock(), gdb_item_mock, zipped_path)

    def test__cleanup_warns_on_file_error(self, mocker):
        expected_warning = "Error deleting zipped gdb /foo/bar/upload.gdb.zip"
        gdb_item_mock = mocker.Mock()
        path_mock = mocker.patch("palletjack.load.Path")
        path_mock.return_value.unlink.side_effect = [IOError]

        with pytest.warns(UserWarning, match=re.escape(expected_warning)):
            load.ServiceUpdater._cleanup(mocker.Mock(), gdb_item_mock, "/foo/bar/upload.gdb.zip")

    def test__cleanup_warns_on_both_agol_and_file_errors(self, mocker):
        expected_agol_warning = "Error deleting gdb item 1234 from AGOL"
        expected_file_warning = f"Error deleting zipped gdb {Path('/foo/bar/upload.gdb.zip')}"
        gdb_item_mock = mocker.Mock(id="1234")
        gdb_item_mock.delete.side_effect = [RuntimeError("Unable to delete item.")]
        zipped_path = Path("/foo/bar/upload.gdb.zip")
        path_mock = mocker.patch("palletjack.load.Path")
        path_mock.return_value.unlink.side_effect = [IOError]

        with pytest.warns(UserWarning) as record:
            load.ServiceUpdater._cleanup(mocker.Mock(), gdb_item_mock, zipped_path)

        assert len(record) == 4  #: Warns again on each with error message
        assert record[0].message.args[0] == expected_agol_warning
        assert record[2].message.args[0] == expected_file_warning


class TestAddToTable:
    def test_add_calls_upsert(self, mocker):
        new_dataframe = pd.DataFrame(
            {
                "foo": [1, 2],
                "bar": [3, 4],
            }
        )
        updater_mock = mocker.Mock(spec=load.ServiceUpdater)
        updater_mock._class_logger = logging.getLogger("mock logger")
        updater_mock.itemid = "foo123"
        updater_mock.service = mocker.Mock()
        updater_mock.service_type = "table"
        updater_mock.index = 0

        updater_mock._update_service.return_value = {"recordCount": 1}

        field_checker_mock = mocker.patch("palletjack.utils.FieldChecker")

        load.ServiceUpdater.add(updater_mock, new_dataframe)

        updater_mock._update_service.assert_called_once_with(
            updater_mock._upload_dataframe.return_value,
            upsert=False,
        )

    def test_add_calls_field_checkers(self, mocker):
        new_dataframe = pd.DataFrame(
            {
                "foo": [1, 2],
                "bar": [3, 4],
            }
        )
        updater_mock = mocker.Mock(spec=load.ServiceUpdater)
        updater_mock.itemid = "foo123"
        updater_mock._class_logger = logging.getLogger("mock logger")
        updater_mock.service_type = "table"
        updater_mock.service = mocker.Mock()
        updater_mock.service.properties.fields = {"a": [1], "b": [2]}
        updater_mock.index = 0

        updater_mock._update_service.return_value = {"recordCount": 1}

        mocker.patch.multiple(
            "palletjack.utils.FieldChecker",
            check_live_and_new_field_types_match=mocker.DEFAULT,
            check_for_non_null_fields=mocker.DEFAULT,
            check_field_length=mocker.DEFAULT,
            check_fields_present=mocker.DEFAULT,
            check_nullable_ints_shapely=mocker.DEFAULT,
        )
        load.ServiceUpdater.add(updater_mock, new_dataframe)

        load.utils.FieldChecker.check_live_and_new_field_types_match.assert_called_once_with(["foo", "bar"])
        load.utils.FieldChecker.check_for_non_null_fields.assert_called_once_with(["foo", "bar"])
        load.utils.FieldChecker.check_field_length.assert_called_once_with(["foo", "bar"])
        load.utils.FieldChecker.check_fields_present.assert_called_once_with(["foo", "bar"], add_oid=False)
        load.utils.FieldChecker.check_nullable_ints_shapely.assert_called_once()
