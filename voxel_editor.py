"""
Stable Enhanced Voxel Editor with hand tracking
Features:
- Color changing voxels
- Bounce animation when drawing
- Stable pinch drawing
- Loading screen
"""

import math
import time
from pathlib import Path
import urllib.request
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------- CONFIG ----------------
GRID_COLS = 14
GRID_ROWS = 7
PINCH_THRESHOLD = 0.045

GRID_COLOR = (224, 216, 105)
BLOCK_EDGE_COLOR = (170, 250, 255)
HIGHLIGHT_FILL_COLOR = (190, 255, 200)
HIGHLIGHT_EDGE_COLOR = (210, 255, 225)

VOXEL_COLORS = [
    (0,255,255),
    (255,0,0),
    (0,255,0),
    (0,128,255),
    (255,255,0)
]

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
HAND_MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

INDEX_TIP_IDX = 8
THUMB_TIP_IDX = 4

# ---------------- DATA CLASSES ----------------

class Cell:
    def __init__(self,x,y):
        self.x=x
        self.y=y

    def __hash__(self):
        return hash((self.x,self.y))

    def __eq__(self,other):
        return self.x==other.x and self.y==other.y


class Voxel:
    def __init__(self,cell,color):
        self.cell=cell
        self.color=color
        self.size_factor=0.1


# ---------------- HAND TRACKING ----------------

def ensure_hand_model():
    if HAND_MODEL_PATH.exists():
        return

    with urllib.request.urlopen(HAND_MODEL_URL) as r:
        HAND_MODEL_PATH.write_bytes(r.read())


class HandTracker:

    def __init__(self):

        ensure_hand_model()

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(HAND_MODEL_PATH)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.last_ts = 0

    def detect(self,frame_rgb):

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        ts = int(time.perf_counter()*1000)

        if ts<=self.last_ts:
            ts=self.last_ts+1

        self.last_ts=ts

        result = self.landmarker.detect_for_video(mp_image,ts)

        if result.hand_landmarks:
            return result.hand_landmarks[0]

        return None

    def close(self):
        self.landmarker.close()


# ---------------- VOXEL EDITOR ----------------

class VoxelEditor:

    def __init__(self):

        self.blocks=[]
        self.highlight=None

        self.frame_w=1
        self.frame_h=1
        self.cell_w=1
        self.cell_h=1

        self.is_pinching=False
        self.pinch_mode=None
        self.last_cell=None

        self.color_index=0

    def update_frame_shape(self,shape):

        self.frame_h,self.frame_w=shape[:2]

        self.cell_w=self.frame_w/GRID_COLS
        self.cell_h=self.frame_h/GRID_ROWS


    def process_landmarks(self,landmarks):

        if landmarks is None:
            self.highlight=None
            self.reset_pinch()
            return

        index_tip = landmarks[INDEX_TIP_IDX]
        thumb_tip = landmarks[THUMB_TIP_IDX]

        cell_x = min(int(index_tip.x*GRID_COLS),GRID_COLS-1)
        cell_y = min(int(index_tip.y*GRID_ROWS),GRID_ROWS-1)

        self.highlight = Cell(cell_x,cell_y)

        dist = math.dist(
            (index_tip.x,index_tip.y),
            (thumb_tip.x,thumb_tip.y)
        )

        if dist < PINCH_THRESHOLD:
            self.handle_pinch()
        else:
            self.reset_pinch()


    def handle_pinch(self):

        if not self.highlight:
            return

        if not self.is_pinching:

            self.is_pinching=True
            self.pinch_mode="erase" if self.cell_exists(self.highlight) else "add"

            self.apply_cell(self.highlight)
            self.last_cell=self.highlight
            return

        if self.highlight != self.last_cell:
            self.apply_cell(self.highlight)
            self.last_cell=self.highlight


    def reset_pinch(self):

        self.is_pinching=False
        self.pinch_mode=None
        self.last_cell=None


    def cell_exists(self,cell):

        return any(v.cell==cell for v in self.blocks)


    def apply_cell(self,cell):

        if self.pinch_mode=="add":

            color = VOXEL_COLORS[self.color_index % len(VOXEL_COLORS)]
            self.color_index+=1

            self.blocks.append(Voxel(cell,color))

        elif self.pinch_mode=="erase":

            for v in self.blocks:
                if v.cell==cell:
                    self.blocks.remove(v)
                    break


    # --------- RENDER ---------

    def render(self,frame):

        canvas=frame.copy()

        self.draw_grid(canvas)
        self.draw_blocks(canvas)

        return canvas


    def draw_grid(self,canvas):

        for c in range(GRID_COLS+1):

            x=int(c*self.cell_w)

            cv2.line(canvas,(x,0),(x,self.frame_h),GRID_COLOR,1)

        for r in range(GRID_ROWS+1):

            y=int(r*self.cell_h)

            cv2.line(canvas,(0,y),(self.frame_w,y),GRID_COLOR,1)

        if self.highlight:

            self.draw_cell(canvas,self.highlight,HIGHLIGHT_FILL_COLOR,filled=True)
            self.draw_cell(canvas,self.highlight,HIGHLIGHT_EDGE_COLOR,thickness=2)


    def draw_blocks(self,canvas):

        for voxel in self.blocks:

            if voxel.size_factor < 1.0:
                voxel.size_factor += 0.2

            self.draw_cell(
                canvas,
                voxel.cell,
                voxel.color,
                filled=True,
                size_factor=voxel.size_factor
            )

            self.draw_cell(canvas,voxel.cell,BLOCK_EDGE_COLOR)


    def draw_cell(self,canvas,cell,color,filled=False,thickness=1,size_factor=1):

        x0=int(cell.x*self.cell_w + (1-size_factor)*self.cell_w/2)
        y0=int(cell.y*self.cell_h + (1-size_factor)*self.cell_h/2)

        x1=int(x0 + self.cell_w*size_factor)
        y1=int(y0 + self.cell_h*size_factor)

        if filled:
            cv2.rectangle(canvas,(x0,y0),(x1,y1),color,cv2.FILLED)

        cv2.rectangle(canvas,(x0,y0),(x1,y1),color,thickness)


# ---------------- MAIN ----------------

def main():

    cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    tracker=HandTracker()
    editor=VoxelEditor()

    cv2.namedWindow("Voxel Editor",cv2.WINDOW_NORMAL)

    # Loading screen
    ret,frame=cap.read()

    if ret:

        loading=frame.copy()

        cv2.putText(
            loading,
            "Loading hand model...",
            (40,60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,255),
            2
        )

        cv2.imshow("Voxel Editor",loading)
        cv2.waitKey(2000)

    try:

        while True:

            ret,frame=cap.read()

            if not ret:
                break

            frame=cv2.flip(frame,1)

            editor.update_frame_shape(frame.shape)

            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            landmarks=tracker.detect(rgb)

            editor.process_landmarks(landmarks)

            output=editor.render(frame)

            cv2.imshow("Voxel Editor",output)

            key=cv2.waitKey(1) & 0xFF

            if key in (27,ord("q")):
                break

    finally:

        tracker.close()

        if cap.isOpened():
            cap.release()

        cv2.destroyAllWindows()

        sys.exit()


if __name__=="__main__":
    main()