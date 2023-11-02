**README.md for image project**

This project contains a Python script for generating SHAP image plots. SHAP image plots are a way to visualize how different features in an image contribute to the prediction of a machine learning model. This code is derived from the SHAP website example usage for image classification at: https://shap.readthedocs.io/en/latest/image_examples.html#image-classification

## To use the project:

1. Install the Conda environment:

```
conda env create -f env.yml
```

2. Activate the Conda environment:

```
conda activate shap-image-plot
```

3. Run the Python script:

```python
python main.py
```

This will generate SHAP image plots for all of the images in the `example_pictures` subfolder. The output images will be saved in the `example_pictures` subfolder as well.

## Example:

```
python main.py
```

This will generate SHAP image plots for all of the images in the `example_pictures` subfolder. The output images will be saved in the `example_pictures` subfolder as well.

## Interpretation of the SHAP image plot:

Each sub-image in the SHAP image plot represents a different feature in the original image. The highlighted parts of the images are the pixels that have the biggest impact on the prediction. The red parts of the images are the pixels that have a positive impact on the prediction, and the blue parts of the images are the pixels that have a negative impact on the prediction.
