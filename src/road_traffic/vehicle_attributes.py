"""Operator-recorded vehicle attributes shared by review validation and search."""

VEHICLE_ATTRIBUTE_VALUES = {
    'color': frozenset({'red','blue','green','yellow','white','black','gray','silver','brown','orange'}),
    'subtype': frozenset({'pickup','semi_trailer','sedan','suv','van','other'}),
}


def validate_vehicle_attributes(attributes, kind):
    if not isinstance(attributes, dict) or not set(attributes) <= set(VEHICLE_ATTRIBUTE_VALUES):
        raise ValueError('Invalid vehicle attributes')
    if attributes and kind not in ('vehicle','candidate'):
        raise ValueError('Invalid vehicle attributes for annotation kind')
    for name,value in attributes.items():
        if not isinstance(value,str) or value not in VEHICLE_ATTRIBUTE_VALUES[name]:
            raise ValueError('Invalid vehicle attributes')
    return dict(attributes)
