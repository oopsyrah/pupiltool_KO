import argparse
import sys
import cv2
import datetime as dt
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

#logger.remove()
#logger.add(sys.stdout, level="WARNING")

INT_E9 = 1_000_000_000


class VideoCapManager:

    def __init__(self, video_path):
        self._path = video_path

    def __enter__(self) -> cv2.VideoCapture:
        self._cap = cv2.VideoCapture(self._path)
        return self._cap

    def __exit__(self, type, value, traceback):
        self._cap.release()


class VideoCapIterator:

    def __init__(self, cap: cv2.VideoCapture, return_timestamp=False):
        self._cap = cap
        self._return_timestamp = return_timestamp

    def __iter__(self):
        return self

    def __next__(self):
        cap = self._cap
        if cap.isOpened():
            frame_exists, curr_frame = cap.read()
            if frame_exists:
                if self._return_timestamp:
                    time_stamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                    return curr_frame, time_stamp
                else:
                    return curr_frame
        raise StopIteration


class VideoCapIndexedIterator:

    def __init__(self,
                 cap: cv2.VideoCapture,
                 return_timestamp=False,
                 progress=False):
        self._cap = cap
        self._return_timestamp = return_timestamp
        self._global_index = 0
        self._local_index = 0
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._progress = progress

    def __iter__(self):
        return self

    def __next__(self):
        cap = self._cap
        if cap.isOpened() and self._global_index < self._total_frames:
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self._global_index)
                frame_exists, curr_frame = cap.read()
            except Exception as e:
                logger.warning(e)
                self._global_index += 1
                return self.__next__()
            else:
                if frame_exists:
                    g_id, l_id = self._global_index, self._local_index
                    rets = (curr_frame, g_id, l_id)
                    if self._return_timestamp:
                        time_stamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                        rets = rets + (time_stamp, )
                    if self._progress:
                        prog = "frame: {} / {}".format(g_id + 1,
                                                       self._total_frames)
                        rets = rets + (prog, )
                    self._global_index += 1
                    self._local_index += 1
                    return rets
        raise StopIteration


def applySIFT(img1,img2):

    sift = cv2.SIFT_create()   
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w,c = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,matrix)
    img3 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    return  kp1,kp2,good,matchesMask,img3,matrix

def get_timestamps(video_path):
    """
    Return:
        timestamps: list of floats, in milliseconds
    """
    with VideoCapManager(video_path) as cap:
        timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
        timestamps = []
        for _, ts in VideoCapIterator(cap, True):
            timestamps.append(ts)
        return timestamps


def pupil_reader(pupil_folder):
    
    video_path = glob.glob(f'{pupil_folder}/*/*.mp4')[0]
    print(video_path)
    event_path = glob.glob(f'{pupil_folder}/*/events.csv')[0]
    gaze_path = glob.glob(f'{pupil_folder}/*/gaze.csv')[0]

    events = pd.read_csv(event_path)
    start = events['timestamp [ns]'].iloc[0] / INT_E9
    start_dt = dt.datetime.fromtimestamp(start)
    timestamps = get_timestamps(video_path)  # in milliseconds

    timestamps_w_offset = []
    for ms in timestamps:
        dts = start_dt + dt.timedelta(milliseconds=ms)
        ts = dt.datetime.timestamp(dts)
        timestamps_w_offset.append(ts)
    frame_time = pd.DataFrame(timestamps_w_offset)

    src_gazes = pd.read_csv(gaze_path)
    cap_mng = VideoCapManager(video_path)

    return cap_mng, frame_time, src_gazes


def gaze_converter(cap_mng, frame_time, src_gazes, stimuli):
    with cap_mng as cap:
        gaze_points = []
        """ 
        fn: global frame number
        vlfn: valid local frame number
        """
        for frame, fn, vlfn, prog in VideoCapIndexedIterator(cap, False, True):
            #logger.info(f"frame #: {fn}, local frame #: {vlfn}")
            logger.info(prog)
            try:
                frame_start = frame_time[0].iloc[vlfn] * INT_E9
                frame_end = frame_time[0].iloc[vlfn + 1] * INT_E9
                frame_segment = src_gazes.loc[
                    (src_gazes['timestamp [ns]'] >= frame_start)
                    & (src_gazes['timestamp [ns]'] < frame_end)]
            except Exception as e1:
                continue
                
            try:
                img1 = stimuli.copy()          
                img2 = frame.copy() 
                kp1,kp2,good,matchesMask,img3,matrix=applySIFT(img1,img2)
                print(len(good))
                if len(good)<15:
                    continue
            except Exception as e2:
                continue

            dst_gazes = []
            for ln in frame_segment.index:
                pic=0
                try:
                    ts_ns, gx, gy, gpx, gpy, worn, fixation_id, blink_id = compute_gaze(
                        ln,
                        frame_segment,
                        src_gazes,
                        matrix, 
                    )
                    img1 = cv2.circle(img1,(int(gpx),int(gpy)),radius=10, color=(255,165,0), thickness=3)
                    img3 = cv2.circle(img3, (int(gx), int(gy)),radius=10, color=(255,165,0), thickness=3)
                    cv2.waitKey(1)
                    dst_gazes.append([fn,ln,ts_ns,gpx,gpy,worn,fixation_id,blink_id,pic])
                except Exception as e4:
                    print(e4)
                    continue
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)
            img4 = cv2.drawMatches(img1,kp1,img3,kp2,good,None,**draw_params)
            cv2.imshow('result',img4)
            
            gaze_points.extend(dst_gazes)
            #frameStatus=0
        return  gaze_points


def compute_gaze(ln, frame_seg, src_gazes, matrix):
    gx = frame_seg['gaze x [px]'].loc[ln]
    gy = frame_seg['gaze y [px]'].loc[ln]
    src_gaze1 = (gx, gy, 1)
    gaze_final_hat = np.linalg.inv(matrix) @ src_gaze1
    gaze_final_hat /= gaze_final_hat[-1]
    (gpx, gpy, gpz) = gaze_final_hat

    ts_ns = src_gazes['timestamp [ns]'].loc[ln]
    worn = src_gazes['worn'].loc[ln]
    fixation_id = src_gazes['fixation id'].loc[ln]
    blink_id = src_gazes['blink id'].loc[ln]
    return ts_ns, gx, gy, gpx, gpy, worn, fixation_id, blink_id


def run():
    trial_folder_src = Path(f'{DATA_FOLDER}/{sub}/{trial}')
    trial_folder_dst = Path(f'{OUTPUT_FOLDER}/{sub}/{trial}')
    if not trial_folder_dst.is_dir():
        trial_folder_dst.mkdir(parents=True)
    #plt.imshow(stimuli)

    """convert gaze"""
    cap_mng, frame_time, src_gazes = pupil_reader(trial_folder_src)
    gaze_points = gaze_converter(cap_mng, frame_time, src_gazes,stimuli)
    pd.DataFrame(gaze_points).to_csv(
        trial_folder_dst / 'gaze_output.csv', index=0)
        


    print(f'{sub} {trial} gaze saved!')


if __name__ == '__main__':
    """
    you should modify the following with your own data
    """
    DATA_FOLDER = Path("./expdata")
    SUB=111801
    TRIAL=1
    STIMPATH=f'{DATA_FOLDER}/stimuli.jpg'
    OUTPUT_FOLDER=f'{DATA_FOLDER}/output'

    """
    run only one trial
    """
    sub=SUB
    trial=TRIAL
    stimuli = cv2.imread(STIMPATH)
    run()
