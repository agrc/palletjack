from era import data_coupler


class TestFeatureServiceInLineUpdater:

    def test_update_feature_service(self, mocker):
        #: We create a mock that will be returned by UpdateCursor's mock's __enter__, thus becoming our context manager.
        #: We then set it's __iter__.return_value to define the data we want it to iterate over.
        contextmanager_mock = mocker.MagicMock()
        contextmanager_mock.__iter__.return_value = [
            ['12345', '42', 123.45, '12/25/2021'],
            ['67890', '18', 67.89, '12/25/2021'],
        ]
        da_mock = mocker.MagicMock()
        da_mock.return_value.__enter__.return_value = contextmanager_mock
        mocker.patch('arcpy.da.UpdateCursor', new=da_mock)

        fsupdater_mock = mocker.Mock()
        fsupdater_mock.data_as_dict = {'12345': {'Count': '57', 'Amount': 100.00, 'Date': '1/1/2022'}}

        data_coupler.FeatureServiceInLineUpdater.update_feature_service(
            fsupdater_mock, 'foo', ['ZipCode', 'Count', 'Amount', 'Date']
        )

        contextmanager_mock.updateRow.assert_called_with(['12345', '57', 100.00, '1/1/2022'])
        contextmanager_mock.updateRow.assert_called_once()
