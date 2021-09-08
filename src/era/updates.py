import json

import numpy as np
import pandas as pd


def calculate_new_stops(dataframe, column, stops=5):
    """Calculate new stop values for an AGOL color ramp

    Args:
        dataframe (pd.DataFrame): Data being classified
        column (str): Column to classify
        stops (int, optional): Number of stops to create. Defaults to 5.

    Returns:
        List: New stops cast as ints
    """

    minval = dataframe[column].min()
    mean = dataframe[column].mean()
    std_dev = dataframe[column].std()
    upper = mean + std_dev  #: AGOL's default upper value for unclassed ramps seems to be mean + 1 std dev

    new_stops = np.linspace(minval, upper, stops)
    new_stops_ints = [int(stop) for stop in new_stops]

    return new_stops_ints


def update_stop_values(webmap_item, layer_number, new_stops):
    """Update the stop values of an (un)classified polygon renderer in an AGOL Webmap

    Args:
        webmap_item (arcgis.gis.Item): The AGOL item of the map to be updated
        layer_number (int): The index for the layer to be updated
        new_stops (List): New values for the existing stops
    """

    #: Get short reference to the stops dictionary from the webmap's data json
    data = webmap_item.get_data()
    renderer = data['operationalLayers'][layer_number]['layerDefinition']['drawingInfo']['renderer']
    stops = renderer['visualVariables'][0]['stops']

    #: Overwrite the value, update the webmap item
    for stop, new_value in zip(stops, new_stops):
        stop['value'] = new_value
    result = webmap_item.update(item_properties={'text': json.dumps(data)})

    return result
