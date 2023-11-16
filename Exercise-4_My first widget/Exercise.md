# Exercise

Here, the process to create a widget:

![DEEPNAPARI_credit(2)](https://github.com/hereariim/DEEP-NAPARI/assets/93375163/c59a1307-87e5-419a-a887-11128700f1fd)

## 1- `_widget.py`

We're going to design a widget to detect nucleus based on deep learning model.
We're going to create a function to contain the script and a GUI.

![DEEPNAPARI_credit](https://github.com/hereariim/DEEP-NAPARI/assets/93375163/9b8c84b4-0c3c-4f48-9bc6-fe394dc1a119)

### Widget Segmentation

We're going to integrate the image processing script (from the notebook script `mifobio.ipynb`) into the code `# SCRIPT`. In , we integrate only code in cell 1-2, not cell 3 because we use napari to display our data. When, you integrate the script, **you should adapt the code at the input and output**.

We have one input:
- image in array

In napari, **image** input is presented as `napari.types` object.

In magicgui, we introduce one variable:
- `selected_image`: current image in napari window which is given by `ImageData` object from `napari.types`.

⚠️Don't forget to import `ImageData` and `LabelsData` in `_widget.py`: `from napari.types import ImageData, LabelsData`

```
from napari.types import ImageData, LabelsData
from napari.viewer import Viewer
import os
import napari_mifobio._paths as paths

@magic_factory(call_button="Run")
def do_model_segmentation(layer: ImageData,image_viewer: Viewer) -> LabelsData:

    model_path_ = os.path.join(paths.get_models_dir(),'ESRF_Seg_Hands_on_best_model.h5')

    model_New = tf.keras.models.load_model(model_path_,custom_objects={'dice_coefficient': dice_coefficient})

    # SCRIPT

    return mask
```

Besides, we need to specify the path of deep learning to run in keras function `load_model`. We use the script `_paths.py` where some functions find themselves the absolute path of the model.

Please, integrate the model `ESRF_Seg_Hands_on_best_model.h5` in `napari_mifobio` folder.

More information about [napari.types](https://napari.org/stable/api/napari.types.html)

⚠️Please, run `pip install -e . ` to update the library

## 2- `napari.yaml`

![DEEPNAPARI_credit(1)](https://github.com/hereariim/DEEP-NAPARI/assets/93375163/68d118a9-9629-47d3-ac5b-4b5e1e742d32)

In contributions section, we add our widget functions:
```
    - id: napari-mifobio.my_widget #must be unique !
      python_name: napari_mifobio:do_model_segmentation
      title: Segmentation
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

![DEEPNAPARI_credit(3)](https://github.com/hereariim/DEEP-NAPARI/assets/93375163/496c2242-799d-4e19-9c89-014ef8cd158e)

To be rigorous, we add our function to the plugin's family of functions
```
__version__ = "0.0.1"
from ._widget import ExampleQWidget, ImageThreshold, threshold_autogenerate_widget, threshold_magic_widget, do_model_segmentation

__all__ = (
    "ExampleQWidget",
    "ImageThreshold",
    "threshold_autogenerate_widget",
    "threshold_magic_widget",
    "do_model_segmentation",
)
```

*See correction: `__init__.py`*

## 4-  `setup.cfg`

![DEEPNAPARI_credit(4)](https://github.com/hereariim/DEEP-NAPARI/assets/93375163/3aaf3301-9ce0-42c2-9f60-3a35cd930dcb)

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

## 5- (Optional for this workshop) `test_widget.py`

![DEEPNAPARI_credit(5)](https://github.com/hereariim/DEEP-NAPARI/assets/93375163/c2ad5495-40ed-44a9-bd96-273a116110ab)

When the plugin is working well, you can add a few tests to each widget to see if the widgets work when a modification has been made to the code. These tests indicate that the plugin is working properly.

Here, we add a test.

Let's suppose a user changes our code.
We add test to check if output is a numpy array and binary

Don't forget to import `do_model_segmentation`


```
from napari_mifobio._widget import (
    ExampleQWidget,
    ImageThreshold,
    threshold_autogenerate_widget,
    threshold_magic_widget,
    do_model_segmentation,
)

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

You can run the test after install pytest (Be at napari-mifobio directory level):
```
pip install ".[testing]"
```
and run the test
```
pytest .
```

*See correction: `test_widget.py`*

## 6-  `README.md`

You should add some relevant information to inform user about your plugin and how to use it.

*See correction: `README.md`*
