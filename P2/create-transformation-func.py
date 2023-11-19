import hopsworks
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
import joblib
import os


# define a new transformation function for having the quality in three brackets, bad <5, 5=< ok <=6, 6< good
def quality_brac(value):
    quality_string = ""
    if value < 5:
        quality_string = "Bad"
    elif (value < 7) and (value > 4):
        quality_string = "Ok"
    else:
        quality_string = "Good"

    return quality_string


if __name__ == '__main__':
    # You have to set the environment variable 'HOPSWORKS_API_KEY' for login to succeed
    project = hopsworks.login()
    fs = project.get_feature_store()

    # create transformation function
    quality_brac_meta = fs.create_transformation_function(
            transformation_function=quality_brac,
            output_type=str,
            version=1
        )
    # persist transformation function in backend
    quality_brac_meta.save()

    # retrieve transformation function
    # quality_brac_fn = fs.get_transformation_function(name="plus_one")

    # delete transformation function from backend
    # plus_one_fn.delete()
