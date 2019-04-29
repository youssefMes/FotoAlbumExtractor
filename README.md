# Photo Album Image Extractor

## Description

This software extracts single photos from scanned photo album pages. For that, all pages should be scanned properly, ideally with a plain and uniform background and no reflections. In case the photos have passepartouts they will be removed in the process. 
Furthermore it is possible to perform a face detection on the extracted photos. The faces will be marked on the photos and extracted as well.
To assess the performance of the photo extraction algorithm, it is possible to compare the results with manually crafted ground truth images.

## Setup for Usage

The project uses:
* Python 3.6.5 
* numpy 1.14.2
* opencv-python 3.4.0

After the Python installation, all dependencies can be installed by using:

```
pip install -r requirements.txt
```

For execution, go to the imextract directory and type:

```
python main.py /path/to/image.tif
```

Further hints for usage and commands can be showed by using: 

```
python main.py -h
```