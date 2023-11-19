# ID2223 Lab 1

## Description
I used modal, hopsworks and huggingface spaces to create a prediction app for wine quality.
The model is fetched from hopsworks, and a user can experiment with wine attributes for a quality prediction through the hugginface + gradio app.

I also wrote an inference script, which makes wine quality predictions on randomly generated data. The inference script is run daily through modal and you can track it through the associated monitoring app.

All code for the wine quality prediction and monitoring are found under `P2/`.

Most code from `P2/` is reworked from the provided code in `P1/`

### Links to the huggingface spaces apps
Prediction App:
https://huggingface.co/spaces/karl-sim/wine

Monitoring App:
https://huggingface.co/spaces/karl-sim/wine-monitor

### Dataset
https://www.kaggle.com/datasets/rajyellow46/wine-quality/data
