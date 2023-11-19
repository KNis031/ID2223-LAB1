import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("wine_model", version=2)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")


def wine(color, fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
         chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph,
         sulphates, alcohol):

    print("Calling function")
    type = 1 if color == 'white' else 0

    df = pd.DataFrame([[type, fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph,
                        sulphates, alcohol]],
                      columns=['type', 'fixed_acidity', "volatile_acidity", "citric_acid", "residual_sugar",
                               "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "ph",
                               "sulphates", "alcohol"])

    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df)
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want
    # the first element.
#     print("Res: {0}").format(res)
    print(res)
    return f"Your wine is {res[0]}!"


demo = gr.Interface(
    fn=wine,
    title="Create your own wine",
    description="Experiment with wine attributes to see how good your wine will be, based on 5295 wine variants!",
    allow_flagging="never",
    inputs=[
        gr.Radio(choices=['white', 'red'], value='white', label="type"),
        gr.Number(value=7.2, label="fixed acidity"),
        gr.Number(value=0.34, label="volatile acidity"),
        gr.Number(value=0.31, label="citric acid"),
        gr.Number(value=5, label="residual sugar"),
        gr.Number(value=0.05, label="chlorides"),
        gr.Number(value=30, label="free sulfur dioxide"),
        gr.Number(value=114, label="total sulfur dioxide"),
        gr.Number(value=0.99, label="density"),
        gr.Number(value=3.2, label="ph"),
        gr.Number(value=0.53, label="sulphates"),
        gr.Number(value=10.5, label="alcohol"),
        ],
    outputs=gr.Textbox()
    )

demo.launch(debug=True)
