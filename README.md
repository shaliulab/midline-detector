# Midline detector

Generate the _ROI_CENTERS.csv file that reports to idocr
where along the x coordinate is the center of the chamber
(or wherever you want the decision zone to be centered around)

Usage:

```
python main.py -i /path/to/idoc/experiment/folder --label # to annotate the center of the ROIs
python main.py -i /path/to/idoc/experiment/folder --render # to visualize an existing annotation
```

## Requirements

```
python>=3.7
opencv-python
pandas
numpy
```
