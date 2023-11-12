# Exercise

## 1- `_widget.py`

We're going to design a widget to detect nucleus based on deep learning model.
We're going to create a function to contain the script and a GUI.

### Widget for thresholding

We have two input:
- image in array

In napari, **image** input is presented as `napari.types` object.

In magicgui, we introduce one variable:
- `selected_image`: current image in napari window which is given by `ImageData` object from `napari.types`.

⚠️Don't forget to import `ImageData` and `LabelsData` in `_widget.py`: `from napari.types import ImageData, LabelsData`

```
from napari.types import ImageData, LabelsData

@magic_factory(call_button="Run")
def do_model_segmentation(layer: ImageData,image_viewer: Viewer) -> LabelsData:
    ###
    SCRIPT
    ###
    return mask
```

More information about [napari.types](https://napari.org/stable/api/napari.types.html)

## 2- `napari.yaml`

In contributions section, we add our widget functions:
```
  - id: napari-mifobio.my_widget #must be unique !
    title: Segmentation
    python_name: napari_mifobio._widget:do_model_segmentation
```
Here, we identify in backend our widget as
```
napari-mifobio.my_widget
```
In widgets section, we add some information to display our widget:
```
  - command: napari-mifobio.my_widget #identity backend
    display_name: Segmentation
```

*See correction: `napari.yaml`*

## 3- `__init__.py`
To be rigorous, we add our function to the plugin's family of functions
```
__version__ = "0.0.1"
from ._widget import ExampleQWidget, example_magic_widget, do_model_segmentation

__all__ = (
    "ExampleQWidget",
    "example_magic_widget",
    "do_model_segmentation",
)
```

*See correction: `__init__.py`*

## 4-  `setup.cfg`
In the configuration file, we specify the libraries we will use in the script to threshold an image. In `[options]` section, we add in `install_requires` variable our used libraries such as `scikit-image`
```
install_requires =
    numpy
    magicgui
    qtpy
    scikit-image
    napari
    tensorflow>=2.11.0
    opencv-python
```

*See correction: `setup.cfg`*

## 5-  `test_widget.py`

When the plugin is working well, you can add a few tests to each widget to see if the widgets work when a modification has been made to the code. These tests indicate that the plugin is working properly.

Here, we add a test.

Let's suppose a user changes our code.
We add test to check if output is a numpy array and binary

```
import pytest
from napari.types import ImageData, LabelsData
from napari.layers import Image, Labels

# We create a RGB image randomly
@pytest.fixture
def im_rgb():
    return ImageData(np.random.randint(256,size=(256,256,3)))

# We establish our function by highlighting the arguments and argument keys (arg of magicgui)
def get_er(*args, **kwargs):
    er_func = do_model_segmentation()
    return er_func(*args, **kwargs)

# We run a test to check if output is numpy array and binary
def test_threshold(im_rgb):
    my_widget_thd = get_er(im_rgb)
    #check if output is numpy array
    assert type(my_widget_thd)==np.ndarray
    #check if output is binary
    assert len(np.unique(my_widget_thd))==2
```

You can run the test after install pytest:
```
pip install ".[testing]" #Be at napari-threshold directory level
```
and run the test
```
pytest .
```

*See correction: `test_widget.py`*

## 6-  `README.md`

You should add some relevant information to inform user about your plugin and how to use it.

*See correction: `README.md`*
