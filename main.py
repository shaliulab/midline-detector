__author__ = "antortjim"
import argparse
import os.path
import re
import sys
import logging

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--experiment-folder", "--input", dest="input", required=True)
ap.add_argument("-D", "--debug", dest="debug", default=False, action="store_true")
args = ap.parse_args()

print(args)

if args.debug:
    logger.setLevel(logging.DEBUG)

def find_file(folder, key):
    ff = os.listdir(folder)
    file = [e  for e in [re.search(key, e) for e in ff] if e is not None][0].string
    file = os.path.join(folder, file)
    return file


def read_snapshots(experiment_folder, n):
    img_snapshots_folder = os.path.join(experiment_folder, "IMG_SNAPSHOTS")
    img_snapshots = [os.path.join(img_snapshots_folder, e) for e in sorted(os.listdir(img_snapshots_folder))]

    snapshots = [cv2.imread(f) for f in img_snapshots[:10]]
    return snapshots

def find_median_image(experiment_folder, roi_mask):

    snapshots = read_snapshots(experiment_folder, 60)

    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in snapshots]
    imgs = [cv2.bitwise_and(img, roi_mask) for img in imgs]

    stack = np.stack(imgs)
    median_image = np.median(stack, axis=0).astype(np.uint8)

    # cv2.imshow("median", median_image)
    return median_image

def get_machine_id(experiment_folder):
    
    metadata_csv=find_file(experiment_folder, "METADATA")
    metadata = pd.read_csv(metadata_csv)
    machine_id = metadata.loc[metadata["field"] == "machine_id"]["value"].squeeze()
    return machine_id

def get_date_time(experiment_folder):

    metadata_csv=find_file(experiment_folder, "METADATA")
    metadata = pd.read_csv(metadata_csv)
    date_time = metadata.loc[metadata["field"] == "date_time"]["value"].squeeze()
    return date_time

def get_roi_map(experiment_folder):
    machine_id = get_machine_id(experiment_folder)
    date_time = get_date_time(experiment_folder)
    
    roi_map_csv = os.path.join(experiment_folder, date_time + "_" + machine_id + "_ROI_MAP.csv")
    roi_map = pd.read_csv(roi_map_csv)
    return roi_map

def make_roi_mask(roi_map, resolution, region_id):

    roi_mask = np.zeros(resolution, dtype=np.uint8)
    x = roi_map.loc[roi_map["value"] == region_id]["x"].squeeze()
    y = roi_map.loc[roi_map["value"] == region_id]["y"].squeeze()
    w = roi_map.loc[roi_map["value"] == region_id]["w"].squeeze()
    h = roi_map.loc[roi_map["value"] == region_id]["h"].squeeze()
    
    roi_mask[y:(y+h), x:(x+w)] = 255
    return roi_mask, (x,y,w,h)


def find_center_human(median, region_id):

    print(f"INFO: Region id {region_id}")
    failed=0
    roi = tuple(cv2.selectROIs("select center", median)[0])
    
    logger.debug("Captured roi: ")
    logger.debug(roi)

    while (len(roi) == 0 or roi == (0,0,0,0)) and failed<2 :
        failed+=1
        roi = tuple(cv2.selectROI("select center", median)[0])

    if failed == 2:
        sys.exit(1)

    x = roi[0]
    return x


def save(experiment_folder, centers):

    machine_id = get_machine_id(experiment_folder)
    date_time = get_date_time(experiment_folder)

    csv_output = os.path.join(
        experiment_folder,
        date_time + "_" + machine_id + "_" + "ROI_CENTER.csv"
    )

    data = ""

    with open(csv_output, "w") as fh:
        header = "region_id,center\n"
        fh.write(header)
        data += header

        for i, c in enumerate(centers):
            new_line = f"{i+1},{c}\n"
            fh.write(new_line)
            data += new_line

    print(data)
    
    return 0

def main():

    experiment_folder = args.input

    roi_map = get_roi_map(experiment_folder)      
    resolution  = read_snapshots(experiment_folder, 1)[0].shape[:2]

    centers = [None, ] * 20

    for region_id in range(1,21):

        roi_mask, coords = make_roi_mask(roi_map, resolution, region_id)
        x,y,w,h=coords
        median = find_median_image(experiment_folder, roi_mask)

        median = median[y:(y+h), x:(x+w)]
        
        increase_factor = 12
        dest_size = tuple(np.array(median.shape[:2])*increase_factor)[::-1]
        median=cv2.resize(median, dest_size, interpolation = cv2.INTER_AREA)
        # center = find_center(median, region_id)
        # center = find_center_canny(median, region_id)
        center = find_center_human(median, region_id)
        center = center / increase_factor + x
        # # print(center)
        centers[region_id-1] = center
        
    save(experiment_folder, centers)


if __name__ == "__main__":

    main()
