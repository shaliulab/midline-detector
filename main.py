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
ap.add_argument("--region-ids", dest="region_ids", nargs="+", default=None, type=int)
ap.add_argument("-l", "--label",  dest="label", action="store_true", default=False)
ap.add_argument("-r", "--render",  dest="render", action="store_true", default=False)
ap.add_argument("-D", "--debug", dest="debug", default=False, action="store_true")
args = ap.parse_args()

print(args)


def draw_arrow(frame, x, y1, y2, *args, color=(0, 0, 255), **kwargs):
    if len(color) == 3 and (len(frame.shape) == 2 or frame.shape[2] == 1):
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    frame = cv2.arrowedLine(frame, (x, y1), (x, y2), *args, color=color, **kwargs)
    return frame


def load_file(experiment_folder, pattern):

    files = os.listdir(experiment_folder)
    files_rev = [f[::-1] for f in files]
    pattern_rev = pattern[::-1]
    file = [f for f in files_rev if f[:len(pattern)] == pattern_rev][0][::-1]
    path = os.path.join(experiment_folder, file)
    result = pd.read_csv(path)
    return result


def render(experiment_folder):
     
    roi_center = load_file(experiment_folder, "ROI_CENTER.csv")
    roi_map = load_file(experiment_folder, "ROI_MAP.csv")
    roi_map["region_id"] = roi_map["value"]
    roi_center.set_index("region_id", inplace=True)
    roi_map.set_index("region_id", inplace=True)
    #import ipdb; ipdb.set_trace()
    roi_center = roi_center.join(roi_map, on="region_id")

    median = find_median_image(experiment_folder)
    print(roi_center)
    
    for roi in range(1, roi_center.shape[0]+1):
        roi_data = roi_center.loc[roi]
        x = int(roi_data["center"])
        print(roi_data)
        y1 = int(roi_data["y"]) - 30
        y2 = int(roi_data["y"]) + 10
        median = draw_arrow(median, x, y1, y2)

    prefix = get_prefix(experiment_folder)
    cv2.imwrite(os.path.join(experiment_folder, prefix + "_" + "MIDLINE_RENDER.png"), median)
    cv2.imshow("roi centers", median)
    cv2.waitKey(0)


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

def find_median_image(experiment_folder, roi_mask = None):

    snapshots = read_snapshots(experiment_folder, 60)

    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in snapshots]
    if roi_mask is None:
        pass
    else:
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
    
    roi = cv2.selectROI("select center", median)
    if roi != (0, 0, 0, 0):
        x = roi[0]
        w = roi[2]
    else:
        return 0

    logger.debug("Captured roi: ")
    logger.debug(roi)

    while roi == (0, 0, 0, 0) and failed < 2 :
        failed+=1
        roi = cv2.selectROI("select center", median)
        if roi != (0, 0, 0, 0):
            x = roi[0]
            w = roi[2]

    if failed == 2:
        sys.exit(1)

    # set the center of the chamber to the center
    # along x of the user selected ROI
    # (and not just the x coordinate of the top left corner)
    x = x + w / 2
    return x


def get_prefix(experiment_folder):
    machine_id = get_machine_id(experiment_folder)
    date_time = get_date_time(experiment_folder)
    prefix = date_time + "_" + machine_id
    return prefix


def save(experiment_folder, centers):

    prefix = get_prefix(experiment_folder)

    csv_output = os.path.join(
        experiment_folder,
        prefix + "_" + "ROI_CENTER.csv"
    )


    if os.path.exists(csv_output):
        centers_dest = pd.read_csv(csv_output)
    else:
        centers_dest = pd.DataFrame({"region_id": list(range(1,21)), "center": [0 for _ in range(1, 21)]})

    for i, c in enumerate(centers):
        if c is None:
            pass
        else:
            centers_dest.loc[centers_dest["region_id"] == i+1, "center"] = c

    centers_dest.to_csv(csv_output, index=False)

#    data = ""
#
#    with open(csv_output, "w") as fh:
#        header = "region_id,center\n"
#        fh.write(header)
#        data += header
#
#        for i, c in enumerate(centers):
#            new_line = f"{i+1},{c}\n"
#            fh.write(new_line)
#            data += new_line
#
#    print(data)
    
    return 0

def label(experiment_folder, region_ids = None):

    roi_map = get_roi_map(experiment_folder)      
    resolution  = read_snapshots(experiment_folder, 1)[0].shape[:2]


    centers = [None, ] * 20
    print(region_ids)
    if region_ids is None:
        region_ids = np.arange(1,21)
    else:
        region_ids = np.array(region_ids)


    for region_id in region_ids:

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
    return 0


def main():

    experiment_folder = args.input

    if args.label:
        label(experiment_folder, args.region_ids)
        return 0

    elif args.render:
        render(experiment_folder)
        return 0
    else:
        print("Please provide --label or --render in CLI")
        return 1

if __name__ == "__main__":

    main()
