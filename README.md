# ML-reflectivity
Fast fitting of reflectivity curves using machine learning. [![DOI](https://zenodo.org/badge/212805647.svg)](https://zenodo.org/badge/latestdoi/212805647)

This code was used in the following scientific publication: 

[1] Fast Fitting of Reflectivity Data of Growing Thin Films Using Neural Networks  A. Greco, V. Starostin, C. Karapanagiotis, A. Hinderhofer, A. Gerlach, L. Pithan, S. Liehr, F. Schreiber, & S. Kowarik (2019). _J. Appl. Cryst._, in print.

A co-developed version of this software with a graphical user interface can be found at [kowarik-labs/AI-reflectivity](https://github.com/kowarik-labs/AI-reflectivity).

## Disclaimer
This repository mainly serves as a public archive for the code used in [1] and is not yet optimized for the use by other researchers. Future updates will improve the functionality and usability of the program.

## Main dependencies
To be able to run the code, a python 3.7 installation with the following dependencies is required:
- keras
- tensorflow
- numpy
- tqdm
- csv
- datetime
- configobj
- matplotlib

## Usage
Provided a correct python installation is available, the code can be run from a suitable shell as-is using the configuration and neural network archtitecture used in [1]. The trained network can be tested on the provided real-time X-ray reflectivity dataset "DIP_330K.txt" of a growing diindenoperylene film on a silicon wafer with a native oxide layer (published in [1]).

#### Generate the reflectivity curves for training
`$ python generate_training_data.py`

#### Define neural network and execute training
`$ python training.py`

#### Predict parameters of the test file "DIP_330K.txt" and plot fitted curves
`$ python prediction.py`

## Authors
- Alessandro Greco (Institut für Angewandte Physik, University of Tübingen)
- Vladimir Starostin (Institut für Angewandte Physik, University of Tübingen)
- Christos Karapanagiotis (Institut für Physik, Humboldt Universität zu Berlin)
- Alexander Hinderhofer (Institut für Angewandte Physik, University of Tübingen)
- Alexander Gerlach (Institut für Angewandte Physik, University of Tübingen)
- Linus Pithan (ESRF The European Synchrotron)
- Sascha Liehr (Bundesanstalt für Materialforschung und -prüfung (BAM))
- Frank Schreiber (Institut für Angewandte Physik, University of Tübingen)
- Stefan Kowarik (Bundesanstalt für Materialforschung und -prüfung (BAM) and Institut für Physik, Humboldt Universität zu Berlin)
