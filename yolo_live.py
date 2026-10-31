from ultralytics import YOLO
import cv2
import math
import time
import threading
import os
import json
import torch
import numpy as np
from penalty_engine import PenaltyEngine
from stable_robot_ids import StableRobotTracker

cv2.setNumThreads(0)
torch.set_num_threads(3)

MODEL_PATH = "models/frc_ref_yolo26s_final.pt"
model = YOLO(MODEL_PATH)
penalties = PenaltyEngine()

source = input("Enter video source. Type \"camera\" or paste YouTube/Twitch URL: ").strip()

if source.lower() == "camera":
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
else:
    from video_stream import VideoStream
    camera = VideoStream(source)

print("FRC START LINE ENGINE v11 - PORTFOLIO VISUALS + CONFIDENCE PAUSE")
print(f"Using model: {MODEL_PATH}")


CONFIDENCE_THRESHOLD = 0.15

MAX_FRAME_WIDTH = 1280
YOLO_IMAGE_SIZE = 640


RUN_YOLO_EVERY_N_FRAMES = 2

BLUE_CLASS = "Robot with blue bumper"
RED_CLASS = "Robot with red bumper"

DETECTION_CLASSES = {
    BLUE_CLASS,
    RED_CLASS
}

print("Loaded classes:")
print(model.names)
print("Detecting:", DETECTION_CLASSES)

WINDOW_NAME = "FRC START LINE ENGINE v11"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


def get_screen_size():
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        root.destroy()

        return screen_width, screen_height

    except Exception:
        return WINDOW_WIDTH, WINDOW_HEIGHT


def resize_to_fill_screen(frame, target_width, target_height):
    frame_height, frame_width = frame.shape[:2]

    scale = max(target_width / frame_width, target_height / frame_height)

    scaled_width = int(frame_width * scale)
    scaled_height = int(frame_height * scale)

    resized = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

    offset_x = max(0, (scaled_width - target_width) // 2)
    offset_y = max(0, (scaled_height - target_height) // 2)

    return resized[offset_y:offset_y + target_height, offset_x:offset_x + target_width]


screen_width, screen_height = get_screen_size()
fullscreen_mode = False

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.moveWindow(WINDOW_NAME, 80, 80)


blue_line_x = None
blue_line_angle = 0

red_line_x = None
red_line_angle = 0

center_line_x = None
center_line_angle = 0

preview_x = 200
preview_angle = 0
stage = "PREVIEW"


blue_on_right = True

START_LINE_TOUCH_MARGIN = 10
CENTER_LINE_CLEAR_MARGIN = 4


state = "WAITING FOR LINES"
override = False
match_phase = "WAITING_FOR_LINES"
phase_start_time = None
phase_durations = {
    "AUTO": 20,
    "WAITING": 4,
    "TRANSITION": 10,
    "SHIFTS": 100,
    "ENDGAME": 30,
}


preview_y = 360


tower_width = 85
tower_height = 140
tower_preview_angle = 0

blue_tower = None
red_tower = None


paused = False
last_frame = None
match_pause_start = None


CONTACT_GAP_PX = 4

CONTACT_BOTTOM_BAND_TOP_RATIO = 0.75
CONTACT_BOTTOM_BAND_BOTTOM_RATIO = 0.95


CONTACT_UPPER_BUMPER_CALIBRATION = [
    (0.45, 0.24, 0.40),
    (0.60, 0.34, 0.52),
    (0.75, 0.45, 0.64),
    (0.92, 0.54, 0.74),
]

CONTACT_BAND_SIDE_SHRINK_RATIO = 0.07
CONTACT_MIN_OVERLAP_RATIO = 0.50
CONTACT_HEIGHT_RATIO_MIN = 0.55

CONTACT_CONFIRM_NEW_RESULTS = 1
CONTACT_MAX_CENTER_SEPARATION_IN = 60.0
contact_pair_hit_streaks = {}
last_contact_result_token = None

ENDGAME_CONTACT_FRAMES_REQUIRED = 3
contact_debug = False
endgame_contact_memory = {}


foul_totals = {
    "BLUE": {
        "MAJOR": 0,
        "MINOR": 0
    },
    "RED": {
        "MAJOR": 0,
        "MINOR": 0
    }
}

foul_log = []
match_reported = False


FOUL_ANNOUNCEMENT_SECONDS = 2.0
foul_announcements = []

current_frame_height = 1


sim_blue = 0
sim_red = 0


show_controls = True
precision_mode = 0

debug_mode = False
debug_robot_label = BLUE_CLASS
debug_robot_width = 90
debug_robot_height = 70
debug_robots = []
selected_debug_robot_index = None
next_debug_robot_tid = -1000


AUTO_CONTACT_FRAMES_REQUIRED = 3
AUTO_CONTACT_RELEASE_FRAMES_REQUIRED = 3
auto_contact_memory = {}


FIELD_LENGTH_IN = 651.2
FIELD_WIDTH_IN = 317.7
FIELD_BOUNDARY_CONTACT_GAP_PX = 4.0
FIELD_CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "field_calibration.json")
field_calibration_points = []
field_homography = None
field_calibration_mode = False


ROBOT_STARTING_LINE_FROM_END_IN = 158.6
TOWER_BASE_DEPTH_IN = 45.18
TOWER_BASE_WIDTH_IN = 39.0


TOWER_WIDTHWISE_FRACTION = 0.50


auto_line_segments = {}
field_auto_setup_enabled = False
selected_field_object = "NONE"
FIELD_EDIT_OBJECTS = (
    "NONE", "BLUE LINE", "CENTER LINE", "RED LINE", "BLUE TOWER", "RED TOWER"
)

PIN_DISTANCE_IN = 72.0
PIN_CLEAR_SECONDS = 3.0
PIN_INITIAL_SECONDS = 3.0
PIN_MAJOR_REPEAT_SECONDS = 3.0
PIN_STATIONARY_WINDOW_SEC = 0.60
PIN_STATIONARY_MAX_TRAVEL_IN = 6.0
PIN_CANDIDATE_CONFIRM_SEC = 0.35
PIN_CANDIDATE_LOST_GRACE_SEC = 0.30
PIN_TRACK_LOST_CLEAR_SEC = 1.0

PIN_CONFIDENCE_THRESHOLD = 0.60

PIN_UNCERTAIN_TIMEOUT_SEC = 3.0
YOLO_STALE_RESULT_SEC = 0.80
robot_field_history = {}
pin_candidate_memory = {}
active_pins = {}


stable_tracker = StableRobotTracker({BLUE_CLASS: 3, RED_CLASS: 3})
last_tracked_yolo_result = None
cached_tracking_ids = []


def reset_track_memory():
    global last_tracked_yolo_result, cached_tracking_ids, last_contact_result_token
    contact_pair_hit_streaks.clear()
    last_contact_result_token = None
    stable_tracker.reset()
    last_tracked_yolo_result = None
    cached_tracking_ids = []

def create_debug_robot(center_x, center_y, label):
    global next_debug_robot_tid

    half_width = debug_robot_width // 2
    half_height = debug_robot_height // 2

    robot = {
        "tid": next_debug_robot_tid,
        "label": label,
        "x1": int(center_x - half_width),
        "y1": int(center_y - half_height),
        "x2": int(center_x + half_width),
        "y2": int(center_y + half_height),
        "cx": int(center_x),
        "cy": int(center_y),
        "debug": True
    }

    next_debug_robot_tid -= 1

    return robot


def move_debug_robot(robot, shift_x, shift_y):
    robot["x1"] += shift_x
    robot["x2"] += shift_x
    robot["y1"] += shift_y
    robot["y2"] += shift_y
    robot["cx"] = (robot["x1"] + robot["x2"]) // 2
    robot["cy"] = (robot["y1"] + robot["y2"]) // 2


def selected_debug_robot():
    if selected_debug_robot_index is None:
        return None

    if selected_debug_robot_index < 0 or selected_debug_robot_index >= len(debug_robots):
        return None

    return debug_robots[selected_debug_robot_index]


def get_move_step():
    if precision_mode == 1:
        return 4
    if precision_mode == 2:
        return 2
    return 8


def get_size_step():
    if precision_mode == 1:
        return 5
    if precision_mode == 2:
        return 2
    return 10


def get_angle_step():
    if precision_mode == 1:
        return 0.5
    if precision_mode == 2:
        return 0.25
    return 1


def resize_debug_robot(robot, width_change, height_change):
    global debug_robot_width
    global debug_robot_height

    if robot is None:
        debug_robot_width = max(20, debug_robot_width + width_change)
        debug_robot_height = max(20, debug_robot_height + height_change)
        print(f"DEBUG ROBOT DEFAULT SIZE: {debug_robot_width}x{debug_robot_height}")
        return

    current_width = max(20, robot["x2"] - robot["x1"])
    current_height = max(20, robot["y2"] - robot["y1"])

    new_width = max(20, current_width + width_change)
    new_height = max(20, current_height + height_change)

    center_x = robot["cx"]
    center_y = robot["cy"]

    half_width = new_width // 2
    half_height = new_height // 2

    robot["x1"] = int(center_x - half_width)
    robot["y1"] = int(center_y - half_height)
    robot["x2"] = int(center_x + half_width)
    robot["y2"] = int(center_y + half_height)
    robot["cx"] = int(center_x)
    robot["cy"] = int(center_y)

    debug_robot_width = new_width
    debug_robot_height = new_height

    print(f"DEBUG ROBOT SIZE: {new_width}x{new_height}")


def report_foul(alliance, severity, description):
    severity = severity.upper()

    if severity not in foul_totals[alliance]:
        foul_totals[alliance][severity] = 0

    foul_totals[alliance][severity] += 1

    entry = {
        "alliance": alliance,
        "severity": severity,
        "description": description,
        "phase": match_phase,
        "time": time.time()
    }

    foul_log.append(entry)


    foul_announcements.append({
        "alliance": alliance,
        "severity": severity,
        "created": time.monotonic(),
    })

    if len(foul_announcements) > 8:
        del foul_announcements[:-8]

    if alliance == "BLUE":
        penalties.add_blue(f"{severity} {description}")
    else:
        penalties.add_red(f"{severity} {description}")

    print(
        f"{alliance} {severity} FOUL: {description} | "
        f"BLUE major={foul_totals['BLUE']['MAJOR']} minor={foul_totals['BLUE']['MINOR']} | "
        f"RED major={foul_totals['RED']['MAJOR']} minor={foul_totals['RED']['MINOR']}"
    )


def draw_foul_announcements(frame):
    now = time.monotonic()
    foul_announcements[:] = [
        item for item in foul_announcements
        if now - item["created"] < FOUL_ANNOUNCEMENT_SECONDS
    ]
    if not foul_announcements:
        return

    height, width = frame.shape[:2]

    current = foul_announcements[-3:]
    banner_width = min(int(width * 0.70), 650)
    banner_height = max(52, int(height * 0.085))
    spacing = max(7, int(height * 0.014))
    top = max(55, int(height * 0.29) - (len(current) - 1) * (banner_height + spacing) // 2)
    left = (width - banner_width) // 2

    for index, event in enumerate(current):
        elapsed = max(0.0, now - event["created"])
        fade = min(1.0, max(0.0, (FOUL_ANNOUNCEMENT_SECONDS - elapsed) / 0.50))

        alliance_color = (255, 120, 45) if event["alliance"] == "BLUE" else (45, 45, 255)
        label = f'{event["alliance"]} {event["severity"]} FOUL'
        y0 = top + index * (banner_height + spacing)
        y1 = y0 + banner_height
        x0, x1 = left, left + banner_width

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (15, 17, 24), -1)
        cv2.addWeighted(overlay, 0.80 * fade, frame, 1.0 - 0.80 * fade, 0, dst=frame)
        cv2.rectangle(frame, (x0, y0), (x1, y1), alliance_color, max(2, round(3 * fade)), cv2.LINE_AA)
        cv2.rectangle(frame, (x0, y0), (x0 + 8, y1), alliance_color, -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.52, min(1.05, width / 1200.0))
        thickness = max(2, int(round(font_scale * 2.5)))
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = max(x0 + 15, (width - text_width) // 2)
        text_y = y0 + (banner_height + text_height) // 2

        cv2.putText(frame, label, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def print_foul_report(title):
    print("")
    print("========== " + title + " ==========")
    print(f"BLUE MAJOR: {foul_totals['BLUE']['MAJOR']}")
    print(f"BLUE MINOR: {foul_totals['BLUE']['MINOR']}")
    print(f"RED MAJOR: {foul_totals['RED']['MAJOR']}")
    print(f"RED MINOR: {foul_totals['RED']['MINOR']}")

    if len(foul_log) == 0:
        print("No fouls recorded.")
    else:
        print("Violations:")
        for index, entry in enumerate(foul_log, start=1):
            print(
                f"{index}. {entry['alliance']} {entry['severity']} "
                f"{entry['description']} during {entry['phase']}"
            )

    print("=================================")
    print("")


def draw_angled_line(frame, line_x, angle_degrees, color, thickness, frame_height):
    angle_rad = math.radians(angle_degrees)


    y1 = 0
    y2 = frame_height


    x_offset_top = -y1 * math.tan(angle_rad)
    x_offset_bottom = -y2 * math.tan(angle_rad)

    start_point = (int(line_x + x_offset_top), int(y1))
    end_point = (int(line_x + x_offset_bottom), int(y2))

    cv2.line(frame, start_point, end_point, color, thickness)

def get_line_x_at_y(y, line_x, angle_degrees):
    angle_rad = math.radians(angle_degrees)
    return line_x - y * math.tan(angle_rad)

def line_side_value(point_x, point_y, line_x, angle_degrees):
    return point_x - get_line_x_at_y(point_y, line_x, angle_degrees)


def box_line_side_values(x1, y1, x2, y2, line_x, angle_degrees):
    return [
        line_side_value(x1, y1, line_x, angle_degrees),
        line_side_value(x2, y1, line_x, angle_degrees),
        line_side_value(x1, y2, line_x, angle_degrees),
        line_side_value(x2, y2, line_x, angle_degrees),
    ]

def box_touches_angled_line(x1, y1, x2, y2, line_x, angle_degrees, margin):
    if line_x is None:
        return False

    values = box_line_side_values(x1, y1, x2, y2, line_x, angle_degrees)

    return min(values) <= margin and max(values) >= -margin

def box_entirely_on_one_side_of_line(x1, y1, x2, y2, line_x, angle_degrees, margin):
    if line_x is None:
        return 0

    values = box_line_side_values(x1, y1, x2, y2, line_x, angle_degrees)

    if min(values) > margin:
        return 1

    if max(values) < -margin:
        return -1

    return 0

def get_side_of_center_line(x, y, line_x, angle_degrees):
    if line_x is None:
        return 0

    line_x_at_y = get_line_x_at_y(y, line_x, angle_degrees)

    if abs(x - line_x_at_y) < 5:
        return 0
    elif x > line_x_at_y:
        return 1
    else:
        return -1


ready_frames = 0

frame_index = 0
last_results = None


yolo_lock = threading.Lock()
yolo_input_frame = None
yolo_input_seq = 0
yolo_output_results = None
yolo_output_seq = -1
yolo_output_timestamp = 0.0
yolo_last_result_time = 0.0
last_yolo_used_seq = -1
yolo_worker_busy = False
yolo_worker_running = True


def yolo_worker_loop():
    global yolo_input_frame
    global yolo_input_seq
    global yolo_output_results
    global yolo_output_seq
    global yolo_output_timestamp
    global yolo_worker_busy

    last_processed_seq = -1

    while yolo_worker_running:
        with yolo_lock:
            if yolo_input_frame is None or yolo_input_seq == last_processed_seq:
                frame_to_process = None
                seq_to_process = None
            else:
                frame_to_process = yolo_input_frame.copy()
                seq_to_process = yolo_input_seq
                yolo_worker_busy = True

        if frame_to_process is None:
            time.sleep(0.005)
            continue

        try:
            yolo_frame = np.ascontiguousarray(frame_to_process)
            results = model(
                yolo_frame,
                imgsz=YOLO_IMAGE_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                device=0,
                verbose=False
            )

            results = [r.cpu() for r in results]

            with yolo_lock:
                yolo_output_results = results
                yolo_output_seq = seq_to_process
                yolo_output_timestamp = time.time()

            last_processed_seq = seq_to_process

        except Exception as error:
            print(f"YOLO worker error: {error}")

        finally:
            with yolo_lock:
                yolo_worker_busy = False


yolo_thread = threading.Thread(target=yolo_worker_loop, daemon=True)
yolo_thread.start()


def robot_touches_rotated_tower(robot, tower, gap):
    tower_polygon = tower_ground_polygon(tower)
    if tower_polygon is None:
        return False

    center = np.mean(tower_polygon, axis=0)
    expanded = center + (tower_polygon - center) * (1.0 + gap / 25.0)
    return any(
        polygons_intersect(zone, expanded.astype(np.float32))
        for zone in get_bumper_zone_polygons(robot).values()
    )


def band_contact_metrics(band_a, band_b):
    ax1, ay1, ax2, ay2 = band_a
    bx1, by1, bx2, by2 = band_b
    aw, bw = max(1, ax2 - ax1), max(1, bx2 - bx1)
    ah, bh = max(1, ay2 - ay1), max(1, by2 - by1)
    horizontal_gap = max(bx1 - ax2, ax1 - bx2, 0)
    vertical_gap = max(by1 - ay2, ay1 - by2, 0)
    horizontal_overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    vertical_overlap = max(0.0, min(ay2, by2) - max(ay1, by1))


    side_ratio = min(1.0, vertical_overlap / min(ah, bh))
    front_ratio = min(1.0, horizontal_overlap / min(aw, bw))
    candidates = []
    if horizontal_gap <= CONTACT_GAP_PX:
        candidates.append(side_ratio)
    if vertical_gap <= CONTACT_GAP_PX:
        candidates.append(front_ratio)
    best_overlap = max(candidates, default=0.0)
    return best_overlap >= CONTACT_MIN_OVERLAP_RATIO, best_overlap


def robot_contact_metrics(robot_a, robot_b):
    height_a = max(1, robot_a["y2"] - robot_a["y1"])
    height_b = max(1, robot_b["y2"] - robot_b["y1"])
    if min(height_a, height_b) / max(height_a, height_b) < CONTACT_HEIGHT_RATIO_MIN:
        return False, 0.0


    if field_homography is not None:
        pos_a = robot_field_center(robot_a)
        pos_b = robot_field_center(robot_b)
        if pos_a is not None and pos_b is not None:
            separation = math.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1])
            if math.isfinite(separation) and separation > CONTACT_MAX_CENTER_SEPARATION_IN:
                return False, 0.0

    a_bottom, a_top = get_bottom_contact_band(robot_a), get_top_contact_band(robot_a)
    b_bottom, b_top = get_bottom_contact_band(robot_b), get_top_contact_band(robot_b)
    pairs = ((a_bottom, b_bottom), (a_top, b_top),
             (a_top, b_bottom), (a_bottom, b_top))
    scores = [band_contact_metrics(a, b) for a, b in pairs]
    passing_scores = [ratio for touching, ratio in scores if touching]
    if passing_scores:
        return True, max(passing_scores)
    return False, max((ratio for _, ratio in scores), default=0.0)


def confirm_contact_pairs(raw_pairs, result_token):
    global last_contact_result_token

    if result_token is not last_contact_result_token:
        next_streaks = {}
        for first, second in raw_pairs:
            if first.get("debug") or second.get("debug"):
                continue
            key = tuple(sorted((first["tid"], second["tid"])))
            next_streaks[key] = min(
                CONTACT_CONFIRM_NEW_RESULTS,
                contact_pair_hit_streaks.get(key, 0) + 1,
            )
        contact_pair_hit_streaks.clear()
        contact_pair_hit_streaks.update(next_streaks)
        last_contact_result_token = result_token

    fresh = inference_is_fresh(time.time())
    return [
        (first, second)
        for first, second in raw_pairs
        if (first.get("debug") or second.get("debug"))
        or (
            fresh and contact_pair_hit_streaks.get(
                tuple(sorted((first["tid"], second["tid"]))), 0
            ) >= CONTACT_CONFIRM_NEW_RESULTS
        )
    ]


def robots_touch_for_foul(robot_a, robot_b):
    return robot_contact_metrics(robot_a, robot_b)[0]


def rect_from_center(center_x, center_y, width, height):
    half_width = width // 2
    half_height = height // 2
    return (
        int(center_x - half_width),
        int(center_y - half_height),
        int(center_x + half_width),
        int(center_y + half_height)
    )

def get_rotated_rect_tuple(center_x, center_y, width, height, angle):
    return (
        (float(center_x), float(center_y)),
        (float(width), float(height)),
        float(angle)
    )

def draw_rotated_rect(frame, center_x, center_y, width, height, angle, color, thickness):
    rect = get_rotated_rect_tuple(center_x, center_y, width, height, angle)
    points = cv2.boxPoints(rect).astype(int)
    cv2.polylines(frame, [points], True, color, thickness)
    return points


def get_upper_bumper_ratios(robot_bottom_x, robot_bottom_y, frame_height):
    points = CONTACT_UPPER_BUMPER_CALIBRATION
    robot_y_ratio = None

    if field_homography is not None:
        mapped = image_points_to_field([(robot_bottom_x, robot_bottom_y)])
        if mapped is not None and len(mapped):
            field_depth = float(mapped[0][1]) / max(1.0, FIELD_WIDTH_IN)
            if math.isfinite(field_depth):
                field_depth = max(0.0, min(1.0, field_depth))
                robot_y_ratio = points[0][0] + field_depth * (points[-1][0] - points[0][0])

    if robot_y_ratio is None:
        robot_y_ratio = robot_bottom_y / max(1, frame_height)

    if robot_y_ratio <= points[0][0]:
        return points[0][1], points[0][2]
    if robot_y_ratio >= points[-1][0]:
        return points[-1][1], points[-1][2]
    for index in range(len(points) - 1):
        y_a, top_a, bottom_a = points[index]
        y_b, top_b, bottom_b = points[index + 1]
        if y_a <= robot_y_ratio <= y_b:
            t = (robot_y_ratio - y_a) / (y_b - y_a)
            return top_a + (top_b - top_a) * t, bottom_a + (bottom_b - bottom_a) * t
    return points[-1][1], points[-1][2]


def get_contact_band(robot, band_type):
    x1 = robot["x1"]
    y1 = robot["y1"]
    x2 = robot["x2"]
    y2 = robot["y2"]

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    side_shrink = int(width * CONTACT_BAND_SIDE_SHRINK_RATIO)

    band_x1 = x1 + side_shrink
    band_x2 = x2 - side_shrink

    if band_type == "top":
        top_ratio, bottom_ratio = get_upper_bumper_ratios(
            (x1 + x2) / 2.0, y2, current_frame_height
        )

        band_y1 = int(y1 + height * top_ratio)
        band_y2 = int(y1 + height * bottom_ratio)

    else:
        band_y1 = int(y1 + height * CONTACT_BOTTOM_BAND_TOP_RATIO)
        band_y2 = int(y1 + height * CONTACT_BOTTOM_BAND_BOTTOM_RATIO)

    if band_x2 <= band_x1:
        band_x1 = x1
        band_x2 = x2

    if band_y2 <= band_y1:
        if band_type == "top":
            band_y1 = int(y1 + height * 0.30)
            band_y2 = int(y1 + height * 0.50)
        else:
            band_y1 = int(y1 + height * 0.80)
            band_y2 = y2

    return (
        band_x1,
        band_y1,
        band_x2,
        band_y2
    )


def get_bottom_contact_band(robot):
    return get_contact_band(robot, "bottom")


def get_top_contact_band(robot):
    return get_contact_band(robot, "top")

def get_bumper_zone_polygons(robot):
    top_x1, top_y1, top_x2, top_y2 = get_top_contact_band(robot)
    bot_x1, bot_y1, bot_x2, bot_y2 = get_bottom_contact_band(robot)

    robot_width = max(1, robot["x2"] - robot["x1"])
    side_width = max(4, int(robot_width * 0.10))

    far_zone = np.array([
        [top_x1, top_y1], [top_x2, top_y1],
        [top_x2, top_y2], [top_x1, top_y2]
    ], dtype=np.float32)

    near_zone = np.array([
        [bot_x1, bot_y1], [bot_x2, bot_y1],
        [bot_x2, bot_y2], [bot_x1, bot_y2]
    ], dtype=np.float32)

    left_outer = float(robot["x1"])
    left_inner = float(min(robot["x2"], robot["x1"] + side_width))
    right_outer = float(robot["x2"])
    right_inner = float(max(robot["x1"], robot["x2"] - side_width))


    side_top = float(top_y2)
    side_bottom = float(max(side_top + 1.0, bot_y1))
    left_zone = np.array([
        [left_outer, side_top], [left_inner, side_top],
        [left_inner, side_bottom], [left_outer, side_bottom]
    ], dtype=np.float32)
    right_zone = np.array([
        [right_inner, side_top], [right_outer, side_top],
        [right_outer, side_bottom], [right_inner, side_bottom]
    ], dtype=np.float32)

    return {
        "FAR": far_zone,
        "NEAR": near_zone,
        "LEFT": left_zone,
        "RIGHT": right_zone,
    }


def get_bumper_footprint_polygon(robot):
    top_x1, top_y1, top_x2, top_y2 = get_top_contact_band(robot)
    bot_x1, bot_y1, bot_x2, bot_y2 = get_bottom_contact_band(robot)

    top_cy = (top_y1 + top_y2) / 2.0
    bot_cy = (bot_y1 + bot_y2) / 2.0

    return np.array([
        [top_x1, top_cy],
        [top_x2, top_cy],
        [bot_x2, bot_cy],
        [bot_x1, bot_cy],
    ], dtype=np.float32)


def compute_field_homography():
    global field_homography

    if len(field_calibration_points) != 4:
        field_homography = None
        return False

    src = np.array(field_calibration_points, dtype=np.float32)
    if not cv2.isContourConvex(src.reshape(-1, 1, 2)) or abs(cv2.contourArea(src)) < 1200:
        field_homography = None
        return False
    dst = np.array([
        [0.0, 0.0],
        [FIELD_LENGTH_IN, 0.0],
        [FIELD_LENGTH_IN, FIELD_WIDTH_IN],
        [0.0, FIELD_WIDTH_IN],
    ], dtype=np.float32)

    field_homography = cv2.getPerspectiveTransform(src, dst)
    return True


def field_points_to_image(field_points):
    if field_homography is None:
        return None
    field_pts = np.asarray(field_points, dtype=np.float32).reshape(-1, 1, 2)
    inv = np.linalg.inv(field_homography)
    mapped = cv2.perspectiveTransform(field_pts, inv)
    if not np.isfinite(mapped).all():
        return None
    return mapped.reshape(-1, 2)


def projected_line_parameters(endpoints):
    (x1, y1), (x2, y2) = np.asarray(endpoints, dtype=float)
    delta_y = y2 - y1
    if abs(delta_y) < 2.0:
        return None
    slope = (x2 - x1) / delta_y
    angle = math.degrees(math.atan(-slope))
    x_at_y0 = x1 - slope * y1
    return float(x_at_y0), float(angle)


def auto_tower_for_end(field_end_x):
    direction = 1.0 if field_end_x < FIELD_LENGTH_IN / 2 else -1.0
    x_wall = field_end_x
    x_inward = field_end_x + direction * TOWER_BASE_DEPTH_IN
    middle_y = FIELD_WIDTH_IN * TOWER_WIDTHWISE_FRACTION
    half_width = TOWER_BASE_WIDTH_IN / 2.0
    footprint = field_points_to_image([
        (x_wall, middle_y - half_width),
        (x_wall, middle_y + half_width),
        (x_inward, middle_y + half_width),
        (x_inward, middle_y - half_width),
    ])
    if footprint is None:
        return None
    center = np.mean(footprint, axis=0)
    return {"x": float(center[0]), "y": float(center[1]),
            "angle": 0.0, "polygon": footprint.astype(float).tolist(),
            "auto": True}


def apply_calibrated_field_setup():
    global blue_line_x, blue_line_angle, red_line_x, red_line_angle
    global center_line_x, center_line_angle, blue_tower, red_tower
    global field_auto_setup_enabled, selected_field_object
    if field_homography is None:
        return False

    ends = field_points_to_image([
        (0.0, FIELD_WIDTH_IN / 2.0),
        (FIELD_LENGTH_IN, FIELD_WIDTH_IN / 2.0),
    ])
    if ends is None or abs(float(ends[0][0] - ends[1][0])) < 15.0:
        print('AUTO SETUP: camera view does not resolve the two field ends.')
        return False


    left_end = 0.0 if ends[0][0] < ends[1][0] else FIELD_LENGTH_IN
    right_end = FIELD_LENGTH_IN if left_end == 0.0 else 0.0
    blue_end = right_end if blue_on_right else left_end
    red_end = left_end if blue_on_right else right_end
    line_positions = {
        'BLUE': (ROBOT_STARTING_LINE_FROM_END_IN if blue_end == 0.0
                 else FIELD_LENGTH_IN - ROBOT_STARTING_LINE_FROM_END_IN),
        'RED': (ROBOT_STARTING_LINE_FROM_END_IN if red_end == 0.0
                else FIELD_LENGTH_IN - ROBOT_STARTING_LINE_FROM_END_IN),
        'CENTER': FIELD_LENGTH_IN / 2.0,
    }
    fitted = {}
    for name, field_x in line_positions.items():
        points = field_points_to_image([
            (field_x, 0.0), (field_x, FIELD_WIDTH_IN)
        ])
        if points is None:
            print(f'AUTO SETUP: could not project {name} line')
            return False
        parameters = projected_line_parameters(points)
        if parameters is None:
            print(f'AUTO SETUP: {name} line is too horizontal for the existing x(y) contact checks.')
            return False
        fitted[name] = (parameters, points.tolist())

    proposed_blue_tower = auto_tower_for_end(blue_end)
    proposed_red_tower = auto_tower_for_end(red_end)
    if proposed_blue_tower is None or proposed_red_tower is None:
        return False

    (blue_line_x, blue_line_angle), auto_line_segments['BLUE'] = fitted['BLUE']
    (red_line_x, red_line_angle), auto_line_segments['RED'] = fitted['RED']
    (center_line_x, center_line_angle), auto_line_segments['CENTER'] = fitted['CENTER']
    blue_tower = proposed_blue_tower
    red_tower = proposed_red_tower
    field_auto_setup_enabled = True
    selected_field_object = 'NONE'
    print('AUTO FIELD SETUP COMPLETE: BLUE / CENTER / RED lines + BLUE / RED tower bases')
    print('Check overlays against the physical tape and tower base plates.')
    print('TAB selects an element; A/D/I/K adjust it; -/= rotate; L exits editing.')
    return True


def draw_projected_or_manual_line(frame, name, line_x, angle, color, height):
    if selected_field_object == f"{name} LINE":
        color = (0, 255, 255)
    if name in auto_line_segments:
        y0 = float(auto_line_segments[name][0][1])
        y1 = float(auto_line_segments[name][1][1])
        p0 = (int(round(get_line_x_at_y(y0, line_x, angle))), int(round(y0)))
        p1 = (int(round(get_line_x_at_y(y1, line_x, angle))), int(round(y1)))
        cv2.line(frame, p0, p1, color, 3, cv2.LINE_AA)
    else:
        draw_angled_line(frame, line_x, angle, color, 3, height)


def tower_ground_polygon(tower):
    if tower is None:
        return None
    if 'polygon' in tower:
        return np.asarray(tower['polygon'], dtype=np.float32)
    rotated_rect = get_rotated_rect_tuple(
        tower['x'], tower['y'], tower_width, tower_height, tower['angle']
    )
    return cv2.boxPoints(rotated_rect).astype(np.float32)


def move_tower_shape(tower, dx, dy, rotate_deg, scale_x, scale_y):
    if tower is None:
        return
    if 'polygon' in tower:
        poly = np.asarray(tower['polygon'], dtype=np.float64)
        center = np.asarray([tower['x'], tower['y']], dtype=float)
        offsets = poly - center
        offsets *= np.asarray([scale_x, scale_y])
        rad = math.radians(rotate_deg)
        rot = np.asarray([[math.cos(rad), -math.sin(rad)],
                          [math.sin(rad), math.cos(rad)]])
        center = center + np.asarray([dx, dy])
        poly = offsets @ rot.T + center
        tower['polygon'] = poly.tolist()
    tower['x'] += dx
    tower['y'] += dy
    tower['angle'] += rotate_deg


def adjust_selected_field_element(dx=0, dy=0, angle_delta=0, scale_x=1, scale_y=1):
    global blue_line_x, red_line_x, center_line_x
    global blue_line_angle, red_line_angle, center_line_angle
    choice = selected_field_object
    if choice == 'BLUE LINE':
        blue_line_x += dx
        blue_line_angle += angle_delta
    elif choice == 'RED LINE':
        red_line_x += dx
        red_line_angle += angle_delta
    elif choice == 'CENTER LINE':
        center_line_x += dx
        center_line_angle += angle_delta
    elif choice == 'BLUE TOWER':
        move_tower_shape(blue_tower, dx, dy, angle_delta, scale_x, scale_y)
    elif choice == 'RED TOWER':
        move_tower_shape(red_tower, dx, dy, angle_delta, scale_x, scale_y)


def swap_calibrated_alliance_objects():
    global blue_line_x, blue_line_angle, red_line_x, red_line_angle
    global blue_tower, red_tower
    blue_line_x, red_line_x = red_line_x, blue_line_x
    blue_line_angle, red_line_angle = red_line_angle, blue_line_angle
    blue_tower, red_tower = red_tower, blue_tower
    old_blue = auto_line_segments.pop('BLUE', None)
    old_red = auto_line_segments.pop('RED', None)
    if old_blue is not None:
        auto_line_segments['RED'] = old_blue
    if old_red is not None:
        auto_line_segments['BLUE'] = old_red


def save_field_calibration():
    payload = {
        "field_length_in": FIELD_LENGTH_IN,
        "field_width_in": FIELD_WIDTH_IN,
        "points": field_calibration_points,
    }
    with open(FIELD_CALIBRATION_FILE, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_field_calibration():
    global field_calibration_points

    if not os.path.exists(FIELD_CALIBRATION_FILE):
        return False

    try:
        with open(FIELD_CALIBRATION_FILE, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        points = payload.get("points", [])
        if len(points) != 4:
            return False

        field_calibration_points = [[float(x), float(y)] for x, y in points]
        if compute_field_homography():
            print(f"Loaded field calibration: {FIELD_CALIBRATION_FILE}")
            apply_calibrated_field_setup()
            return True
    except Exception as error:
        print(f"Could not load field calibration: {error}")

    return False


def field_mouse_callback(event, x, y, _flags, _userdata):
    global field_calibration_mode
    global field_calibration_points

    if event != cv2.EVENT_LBUTTONDOWN or not field_calibration_mode:
        return

    field_calibration_points.append([float(x), float(y)])
    print(f"FIELD CORNER {len(field_calibration_points)}: ({x}, {y})")

    if len(field_calibration_points) == 4:
        if compute_field_homography():
            save_field_calibration()
            field_calibration_mode = False
            print("FIELD CALIBRATION COMPLETE")
            apply_calibrated_field_setup()
            print(f"Saved to: {FIELD_CALIBRATION_FILE}")
        else:
            print("FIELD CALIBRATION FAILED. Press ; and try again.")
            field_calibration_points = []


def image_points_to_field(points):
    if field_homography is None:
        return None

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, field_homography)
    return transformed.reshape(-1, 2)


def robot_field_polygon(robot):
    return image_points_to_field(get_bumper_footprint_polygon(robot))


def robot_field_center(robot):
    polygon = robot_field_polygon(robot)
    if polygon is None:
        return None
    center = np.mean(polygon, axis=0)
    return float(center[0]), float(center[1])


def segment_contact_polygon(point_a, point_b, gap_px):
    ax, ay = float(point_a[0]), float(point_a[1])
    bx, by = float(point_b[0]), float(point_b[1])
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1.0:
        return None
    rect = (
        ((ax + bx) / 2.0, (ay + by) / 2.0),
        (length, max(1.0, gap_px * 2.0 + 1.0)),
        math.degrees(math.atan2(dy, dx)),
    )
    return cv2.boxPoints(rect).astype(np.float32)


def polygons_intersect(poly_a, poly_b):
    if poly_a is None or poly_b is None:
        return False
    try:
        area, _ = cv2.intersectConvexConvex(
            np.asarray(poly_a, dtype=np.float32),
            np.asarray(poly_b, dtype=np.float32),
        )
        return area > 1e-6
    except cv2.error:
        return False


def field_boundary_zones_for_robot(robot):
    if len(field_calibration_points) != 4 or field_homography is None:
        return []

    boundaries = []
    for index in range(4):
        segment = segment_contact_polygon(
            field_calibration_points[index],
            field_calibration_points[(index + 1) % 4],
            FIELD_BOUNDARY_CONTACT_GAP_PX,
        )
        if segment is not None:
            boundaries.append(segment)

    return [
        name
        for name, zone in get_bumper_zone_polygons(robot).items()
        if any(polygons_intersect(zone, boundary) for boundary in boundaries)
    ]


def point_to_segment_distance(point, seg_a, seg_b):
    point_x, point_y = float(point[0]), float(point[1])
    start_x, start_y = float(seg_a[0]), float(seg_a[1])
    end_x, end_y = float(seg_b[0]), float(seg_b[1])

    dx = end_x - start_x
    dy = end_y - start_y
    length_sq = dx * dx + dy * dy

    if length_sq <= 1e-9:
        return math.hypot(point_x - start_x, point_y - start_y)

    t = ((point_x - start_x) * dx + (point_y - start_y) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    nearest_x = start_x + t * dx
    nearest_y = start_y + t * dy
    return math.hypot(point_x - nearest_x, point_y - nearest_y)


def polygons_overlap(poly_a, poly_b):
    contour_a = np.asarray(poly_a, dtype=np.float32).reshape(-1, 2)
    contour_b = np.asarray(poly_b, dtype=np.float32).reshape(-1, 2)

    try:
        area, _ = cv2.intersectConvexConvex(contour_a, contour_b)
        return area > 1e-6
    except cv2.error:
        return False


def polygon_min_distance(poly_a, poly_b):
    if poly_a is None or poly_b is None:
        return None

    if polygons_overlap(poly_a, poly_b):
        return 0.0

    best = float("inf")
    for point in poly_a:
        for i in range(len(poly_b)):
            best = min(best, point_to_segment_distance(point, poly_b[i], poly_b[(i + 1) % len(poly_b)]))

    for point in poly_b:
        for i in range(len(poly_a)):
            best = min(best, point_to_segment_distance(point, poly_a[i], poly_a[(i + 1) % len(poly_a)]))

    return best


def update_robot_field_history(robots, now):
    if field_homography is None:
        return

    for robot in robots:
        position = robot_field_center(robot)
        if position is None:
            continue

        history = robot_field_history.setdefault(robot["tid"], [])
        history.append((now, position[0], position[1]))
        cutoff = now - 3.0
        while history and history[0][0] < cutoff:
            history.pop(0)


def history_position_near(robot_id, target_time):
    history = robot_field_history.get(robot_id, [])
    if not history:
        return None
    sample = min(history, key=lambda item: abs(item[0] - target_time))
    return float(sample[1]), float(sample[2])


def robot_stationary_for_pin(robot_id, now):
    history = robot_field_history.get(robot_id, [])
    if len(history) < 2:
        return False

    cutoff = now - PIN_STATIONARY_WINDOW_SEC
    recent = [sample for sample in history if sample[0] >= cutoff]
    if len(recent) < 2:
        return False

    if recent[0][0] > cutoff + 0.15:
        return False

    latest = recent[-1]
    max_distance = max(
        math.hypot(sample[1] - latest[1], sample[2] - latest[2])
        for sample in recent
    )
    return max_distance <= PIN_STATIONARY_MAX_TRAVEL_IN


def alliance_from_label(label):
    return "BLUE" if label == BLUE_CLASS else "RED"


def robot_completely_across_center(robot):
    if center_line_x is None:
        return False

    side = box_entirely_on_one_side_of_line(
        robot["x1"], robot["y1"], robot["x2"], robot["y2"],
        center_line_x, center_line_angle, CENTER_LINE_CLEAR_MARGIN
    )
    if side == 0:
        return False

    if robot["label"] == BLUE_CLASS:
        own_side = 1 if blue_on_right else -1
    else:
        own_side = -1 if blue_on_right else 1

    return side == -own_side


def update_auto_contact_fouls(opposing_contact_pairs):
    if not in_auto:
        auto_contact_memory.clear()
        return

    seen_pair_keys = set()

    for robot_a, robot_b in opposing_contact_pairs:
        pair_key = tuple(sorted((robot_a["tid"], robot_b["tid"])))
        seen_pair_keys.add(pair_key)

        data = auto_contact_memory.setdefault(pair_key, {
            "contact_frames": 0,
            "release_frames": 0,
            "fouled_offenders": set(),
        })

        data["contact_frames"] += 1
        data["release_frames"] = 0

        if data["contact_frames"] < AUTO_CONTACT_FRAMES_REQUIRED:
            continue

        for offender in (robot_a, robot_b):
            if offender["tid"] in data["fouled_offenders"]:
                continue
            if not robot_completely_across_center(offender):
                continue

            report_foul(
                alliance_from_label(offender["label"]),
                "MAJOR",
                "AUTO CROSS-LINE CONTACT"
            )
            data["fouled_offenders"].add(offender["tid"])

    for pair_key in list(auto_contact_memory.keys()):
        if pair_key in seen_pair_keys:
            continue

        data = auto_contact_memory[pair_key]
        data["release_frames"] += 1

        if data["release_frames"] >= AUTO_CONTACT_RELEASE_FRAMES_REQUIRED:
            auto_contact_memory.pop(pair_key, None)


def activate_pin(pinner, pinned, start_time, now):
    key = (pinner["tid"], pinned["tid"])
    if key in active_pins:
        return

    pinner_start = history_position_near(pinner["tid"], start_time) or robot_field_center(pinner)
    pinned_start = history_position_near(pinned["tid"], start_time) or robot_field_center(pinned)

    if pinner_start is None or pinned_start is None:
        return

    active_pins[key] = {
        "pinner_tid": pinner["tid"],
        "pinned_tid": pinned["tid"],
        "pinner_label": pinner["label"],
        "start_time": start_time,
        "pinner_start": pinner_start,
        "pinned_start": pinned_start,
        "count_seconds": max(0.0, now - start_time),
        "last_update": now,
        "a_since": None,
        "b_since": None,
        "minor_given": False,
        "majors_given": 0,
        "missing_since": None,
        "uncertain_since": None,
        "pause_reason": None,
    }
    print(f"G418 PIN START: {alliance_from_label(pinner['label'])} TID {pinner['tid']} pinning TID {pinned['tid']}")


def robot_detection_confidence(robot):

    return 1.0 if robot.get("debug") else float(robot.get("confidence", 0.0))


def freeze_pin_clock(pin, now, reason):
    dt = max(0.0, now - pin["last_update"])
    pin["last_update"] = now
    if pin.get("a_since") is not None:
        pin["a_since"] += dt
    if pin.get("b_since") is not None:
        pin["b_since"] += dt
    pin["pause_reason"] = reason
    if pin.get("uncertain_since") is None:
        pin["uncertain_since"] = now


def inference_is_fresh(now):
    return yolo_last_result_time > 0 and 0 <= now - yolo_last_result_time <= YOLO_STALE_RESULT_SEC


def update_pin_detection(robots, opposing_contact_pairs, now):
    pin_rule_active = match_phase in {"AUTO", "TRANSITION", "SHIFTS", "ENDGAME"}

    if field_homography is None or not pin_rule_active:
        pin_candidate_memory.clear()
        active_pins.clear()
        return

    robots_by_tid = {robot["tid"]: robot for robot in robots}
    update_robot_field_history(robots, now)

    seen_candidates = set()

    for robot_a, robot_b in opposing_contact_pairs:
        if not (robot_a.get("debug") and robot_b.get("debug")) and not inference_is_fresh(now):
            continue
        if min(robot_detection_confidence(robot_a), robot_detection_confidence(robot_b)) < PIN_CONFIDENCE_THRESHOLD:
            continue
        a_boundary = field_boundary_zones_for_robot(robot_a)
        b_boundary = field_boundary_zones_for_robot(robot_b)


        if bool(a_boundary) == bool(b_boundary):
            continue

        if a_boundary:
            pinned = robot_a
            pinner = robot_b
        else:
            pinned = robot_b
            pinner = robot_a

        if not robot_stationary_for_pin(pinned["tid"], now):
            continue

        key = (pinner["tid"], pinned["tid"])
        seen_candidates.add(key)

        candidate = pin_candidate_memory.get(key)
        if candidate is None:
            candidate = {
                "since": now,
                "last_seen": now,
            }
            pin_candidate_memory[key] = candidate
        else:
            candidate["last_seen"] = now

        if key not in active_pins and now - candidate["since"] >= PIN_CANDIDATE_CONFIRM_SEC:
            activate_pin(pinner, pinned, candidate["since"], now)

    for key in list(pin_candidate_memory.keys()):
        if key in seen_candidates:
            continue
        if now - pin_candidate_memory[key].get("last_seen", 0) > PIN_CANDIDATE_LOST_GRACE_SEC:
            pin_candidate_memory.pop(key, None)

    ended_keys = set()

    for key, pin in list(active_pins.items()):
        pinner = robots_by_tid.get(pin["pinner_tid"])
        pinned = robots_by_tid.get(pin["pinned_tid"])

        if pinner is None or pinned is None:
            if pin["missing_since"] is None:
                pin["missing_since"] = now
            freeze_pin_clock(pin, now, "NO TRACK")
            if now - pin["missing_since"] > PIN_TRACK_LOST_CLEAR_SEC:
                print(f"G418 PIN DROPPED: robot track missing for {PIN_TRACK_LOST_CLEAR_SEC:.1f}s")
                ended_keys.add(key)
            continue

        pin["missing_since"] = None

        reliable = (
            (pinner.get("debug") and pinned.get("debug"))
            or inference_is_fresh(now)
        )
        if not reliable:
            freeze_pin_clock(pin, now, "STALE YOLO")
        elif min(robot_detection_confidence(pinner), robot_detection_confidence(pinned)) < PIN_CONFIDENCE_THRESHOLD:
            freeze_pin_clock(pin, now, "LOW CONF")
        else:
            pin["pause_reason"] = None
            pin["uncertain_since"] = None

        if pin["pause_reason"] is not None:
            if now - pin["uncertain_since"] > PIN_UNCERTAIN_TIMEOUT_SEC:
                print("G418 PIN DROPPED: detections unreliable too long")
                ended_keys.add(key)
            continue

        pinner_center = robot_field_center(pinner)
        pinned_center = robot_field_center(pinned)
        pinner_poly = robot_field_polygon(pinner)
        pinned_poly = robot_field_polygon(pinned)

        if pinner_center is None or pinned_center is None or pinner_poly is None or pinned_poly is None:
            pin["last_update"] = now
            continue

        robot_gap = polygon_min_distance(pinner_poly, pinned_poly)
        criterion_a = robot_gap is not None and robot_gap >= PIN_DISTANCE_IN

        pinner_moved = math.hypot(
            pinner_center[0] - pin["pinner_start"][0],
            pinner_center[1] - pin["pinner_start"][1]
        )
        pinned_moved = math.hypot(
            pinned_center[0] - pin["pinned_start"][0],
            pinned_center[1] - pin["pinned_start"][1]
        )
        criterion_b = pinner_moved >= PIN_DISTANCE_IN or pinned_moved >= PIN_DISTANCE_IN

        if criterion_a:
            if pin["a_since"] is None:
                pin["a_since"] = now
        else:
            pin["a_since"] = None

        if criterion_b:
            if pin["b_since"] is None:
                pin["b_since"] = now
        else:
            pin["b_since"] = None

        if pin["a_since"] is not None and now - pin["a_since"] > PIN_CLEAR_SECONDS:
            print("G418 PIN CLEARED: criterion A")
            ended_keys.add(key)
            continue

        if pin["b_since"] is not None and now - pin["b_since"] > PIN_CLEAR_SECONDS:
            print("G418 PIN CLEARED: criterion B")
            ended_keys.add(key)
            continue

        dt = max(0.0, now - pin["last_update"])
        pin["last_update"] = now


        if not criterion_a and not criterion_b:
            pin["count_seconds"] += dt

        if pin["count_seconds"] >= PIN_INITIAL_SECONDS and not pin["minor_given"]:
            report_foul(
                alliance_from_label(pin["pinner_label"]),
                "MINOR",
                "G418 PIN"
            )
            pin["minor_given"] = True

        expected_majors = 0
        if pin["count_seconds"] >= PIN_INITIAL_SECONDS + PIN_MAJOR_REPEAT_SECONDS:
            expected_majors = int(
                (pin["count_seconds"] - PIN_INITIAL_SECONDS) // PIN_MAJOR_REPEAT_SECONDS
            )

        while pin["majors_given"] < expected_majors:
            report_foul(
                alliance_from_label(pin["pinner_label"]),
                "MAJOR",
                "G418 EXTENDED PIN"
            )
            pin["majors_given"] += 1


    pinned_tids = {
        pin["pinned_tid"]
        for key, pin in active_pins.items()
        if key not in ended_keys
    }
    for key, pin in active_pins.items():
        if key in ended_keys:
            continue
        if pin["pinner_tid"] in pinned_tids:
            print("G418 PIN CLEARED: criterion C")
            ended_keys.add(key)

    for key in ended_keys:
        active_pins.pop(key, None)
        pin_candidate_memory.pop(key, None)


def draw_field_and_pin_debug(frame):
    if len(field_calibration_points) > 0:
        pts = np.array(field_calibration_points, dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(frame, [pts], len(pts) == 4, (0, 255, 0), 2)
        for index, point in enumerate(pts, start=1):
            cv2.circle(frame, tuple(point), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(index), tuple(point + np.array([6, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    if field_calibration_mode:
        cv2.putText(frame, f"FIELD CALIBRATION: click corner {len(field_calibration_points) + 1}/4", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    y = 235
    for pin in active_pins.values():
        pause = ""
        if pin.get("pause_reason"):
            pause += " PAUSED-" + pin["pause_reason"]
        if pin.get("a_since") is not None:
            pause += " PAUSED-A"
        if pin.get("b_since") is not None:
            pause += " PAUSED-B"
        text = f"PIN {alliance_from_label(pin['pinner_label'])} {pin['count_seconds']:.1f}s{pause}"
        cv2.putText(frame, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)
        y += 22

def draw_contact_debug_overlay(frame, robots, contact_pairs):
    for robot in robots:
        zones = get_bumper_zone_polygons(robot)
        boundary_zones = set(field_boundary_zones_for_robot(robot))

        for zone_name, polygon in zones.items():
            pts = polygon.astype(np.int32)
            if zone_name in boundary_zones:
                color = (255, 255, 255)
                thickness = 3
            elif zone_name == "FAR":
                color = (0, 255, 255)
                thickness = 1
            elif zone_name == "NEAR":
                color = (255, 0, 255)
                thickness = 1
            elif zone_name == "LEFT":
                color = (0, 255, 0)
                thickness = 1
            else:
                color = (255, 255, 0)
                thickness = 1

            cv2.polylines(frame, [pts], True, color, thickness)
            centroid = np.mean(pts, axis=0).astype(int)
            cv2.putText(frame, zone_name[0], tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        if boundary_zones:
            cv2.putText(
                frame,
                "FIELD:" + ",".join(sorted(boundary_zones)),
                (robot["x1"], max(15, robot["y1"] - 28)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                2
            )


        if field_homography is not None:
            center = robot_field_center(robot)
            if center is not None and all(math.isfinite(v) for v in center):
                cv2.putText(
                    frame, f"EST FIELD X:{center[0]:.0f}in Y:{center[1]:.0f}in",
                    (robot["x1"], min(frame.shape[0] - 12, robot["y2"] + 17)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
                )

    for robot_a, robot_b in contact_pairs:
        ax = int((robot_a["cx"] + robot_b["cx"]) / 2)
        ay = int((robot_a["cy"] + robot_b["cy"]) / 2)

        cv2.line(
            frame,
            (robot_a["cx"], robot_a["cy"]),
            (robot_b["cx"], robot_b["cy"]),
            (255, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"CONTACT edge overlap:{robot_contact_metrics(robot_a, robot_b)[1] * 100:.0f}%",
            (ax, ay),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            2
        )

def contact_confirmed(contact_key):
    now = time.time()

    data = endgame_contact_memory.get(
        contact_key,
        {
            "count": 0,
            "last_seen": 0,
            "last_foul": 0
        }
    )

    if now - data["last_seen"] > 0.35:
        data["count"] = 0

    data["count"] += 1
    data["last_seen"] = now

    endgame_contact_memory[contact_key] = data

    return data["count"] >= ENDGAME_CONTACT_FRAMES_REQUIRED

def get_alliance_at_point(x, y, center_x, center_angle, blue_right):
    if center_x is None:
        return "BLUE" if blue_right else "RED"

    side = get_side_of_center_line(x, y, center_x, center_angle)

    if side == 0:
        side = 1 if blue_right else -1

    if (side == 1 and blue_right) or (side == -1 and not blue_right):
        return "BLUE"

    return "RED"

cv2.setMouseCallback(WINDOW_NAME, field_mouse_callback)
load_field_calibration()

window_started = False
while True:
    if not paused or last_frame is None:
        ret, frame = camera.read()
        if not ret:
            break

        if source.lower() == "camera":
            frame = cv2.flip(frame, 1)

        last_frame = frame.copy()
    else:
        frame = last_frame.copy()

    if frame.shape[1] > MAX_FRAME_WIDTH:
        scale = MAX_FRAME_WIDTH / frame.shape[1]
        new_width = int(frame.shape[1] * scale)
        new_height = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, (new_width, new_height))

    h, w, _ = frame.shape
    current_frame_height = h
    annotated = frame.copy()

    frame_index += 1

    if last_results is None:

        yolo_frame = np.ascontiguousarray(frame)
        last_results = model(
            yolo_frame,
            imgsz=YOLO_IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            device=0,
            verbose=False
        )
        last_results = [r.cpu() for r in last_results]
        yolo_last_result_time = time.time()
        results = last_results

    else:


        if frame_index % RUN_YOLO_EVERY_N_FRAMES == 0:
            with yolo_lock:
                if not yolo_worker_busy:
                    yolo_input_frame = frame.copy()
                    yolo_input_seq += 1


        with yolo_lock:
            if yolo_output_results is not None and yolo_output_seq != last_yolo_used_seq:
                last_results = yolo_output_results
                yolo_last_result_time = yolo_output_timestamp
                last_yolo_used_seq = yolo_output_seq

        results = last_results

    penalties.clear_frame()

    blue_touch_count = 0
    red_touch_count = 0


    in_auto = False
    in_waiting = False
    in_transition = False
    in_shifts = False
    in_endgame = False
    phase_time_remaining = 0

    if phase_start_time is not None and match_phase in phase_durations:
        match_now = time.time()

        if paused and match_pause_start is not None:
            match_now = match_pause_start

        elapsed = match_now - phase_start_time
        duration = phase_durations.get(match_phase, 0)
        phase_time_remaining = max(0.0, duration - elapsed)

        if match_phase == "AUTO":
            if elapsed < duration:
                in_auto = True
            else:
                match_phase = "WAITING"
                phase_start_time = time.time()

        elif match_phase == "WAITING":
            if elapsed < duration:
                in_waiting = True
            else:
                match_phase = "TRANSITION"
                phase_start_time = time.time()

        elif match_phase == "TRANSITION":
            if elapsed < duration:
                in_transition = True
            else:
                match_phase = "SHIFTS"
                phase_start_time = time.time()

        elif match_phase == "SHIFTS":
            if elapsed < duration:
                in_shifts = True
            else:
                match_phase = "ENDGAME"
                phase_start_time = time.time()

        elif match_phase == "ENDGAME":
            if elapsed < duration:
                in_endgame = True
            else:

                match_phase = "MATCH_OVER"
                phase_start_time = None
                state = ""

                if not match_reported:
                    print_foul_report("FINAL FOUL REPORT")
                    match_reported = True

    contact_debug_pairs = []


    robots = []
    robot_color_map = {}


    tracking_candidates = []
    for candidate_box in results[0].boxes:
        candidate_class = int(candidate_box.cls[0])
        candidate_label = model.names[candidate_class]
        candidate_conf = float(candidate_box.conf[0])
        if candidate_conf < CONFIDENCE_THRESHOLD or candidate_label not in DETECTION_CLASSES:
            continue
        tx1, ty1, tx2, ty2 = map(int, candidate_box.xyxy[0])
        tracking_candidates.append({
            "label": candidate_label, "confidence": candidate_conf,
            "x1": tx1, "y1": ty1, "x2": tx2, "y2": ty2,
            "cx": (tx1 + tx2) // 2, "cy": (ty1 + ty2) // 2
        })
    if results is not last_tracked_yolo_result:
        cached_tracking_ids = stable_tracker.update(tracking_candidates, frame_index)
        last_tracked_yolo_result = results
    id_for_next_detection = iter(cached_tracking_ids)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        confidence = float(box.conf[0])

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        if label not in DETECTION_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        tid = next(id_for_next_detection)


        if tid is None:
            raise RuntimeError("No-drop tracker returned None for a YOLO detection")
        robot_color_map[tid] = label

        robots.append({
            "tid": tid, "label": label, "confidence": confidence,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": cx, "cy": cy,
        })

        touching_blue = False
        touching_red = False

        if (
            label == BLUE_CLASS
            and blue_line_x is not None
            and box_touches_angled_line(x1, y1, x2, y2, blue_line_x, blue_line_angle, START_LINE_TOUCH_MARGIN)
        ):
            touching_blue = True
            blue_touch_count += 1

        if (
            label == RED_CLASS
            and red_line_x is not None
            and box_touches_angled_line(x1, y1, x2, y2, red_line_x, red_line_angle, START_LINE_TOUCH_MARGIN)
        ):
            touching_red = True
            red_touch_count += 1


        if label == BLUE_CLASS:
            color = (255, 0, 0)
        elif label == RED_CLASS:
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            annotated, f"{'BLUE' if label == BLUE_CLASS else 'RED'} ID:{tid} conf:{confidence:.2f}",
            (x1, max(13, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 2
        )

    for debug_robot in debug_robots:
        robots.append(debug_robot)

        label = debug_robot["label"]
        x1 = debug_robot["x1"]
        y1 = debug_robot["y1"]
        x2 = debug_robot["x2"]
        y2 = debug_robot["y2"]

        if label == BLUE_CLASS:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)

        if (
            label == BLUE_CLASS
            and blue_line_x is not None
            and box_touches_angled_line(x1, y1, x2, y2, blue_line_x, blue_line_angle, START_LINE_TOUCH_MARGIN)
        ):
            blue_touch_count += 1

        if (
            label == RED_CLASS
            and red_line_x is not None
            and box_touches_angled_line(x1, y1, x2, y2, red_line_x, red_line_angle, START_LINE_TOUCH_MARGIN)
        ):
            red_touch_count += 1

        thickness = 3 if debug_robot is selected_debug_robot() else 2

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            annotated,
            f"DEBUG {'BLUE' if label == BLUE_CLASS else 'RED'} ID:{debug_robot['tid']}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            2
        )


    opposing_contact_pairs = []
    for index_a in range(len(robots)):
        for index_b in range(index_a + 1, len(robots)):
            robot_a = robots[index_a]
            robot_b = robots[index_b]

            if robot_a["label"] == robot_b["label"]:
                continue

            if robots_touch_for_foul(robot_a, robot_b):
                opposing_contact_pairs.append((robot_a, robot_b))


    opposing_contact_pairs = confirm_contact_pairs(opposing_contact_pairs, results)
    contact_debug_pairs = list(opposing_contact_pairs)


    update_auto_contact_fouls(opposing_contact_pairs)


    logic_now = match_pause_start if paused and match_pause_start is not None else time.time()
    update_pin_detection(robots, opposing_contact_pairs, logic_now)


    if in_endgame:
        tower_list = [
            ("BLUE", blue_tower),
            ("RED", red_tower)
        ]

        for tower_alliance, tower in tower_list:
            if tower is None:
                continue

            tower_rect = rect_from_center(
                tower["x"],
                tower["y"],
                tower_width,
                tower_height
            )

            protected_class = BLUE_CLASS if tower_alliance == "BLUE" else RED_CLASS
            opponent_class = RED_CLASS if tower_alliance == "BLUE" else BLUE_CLASS

            for protected_robot in robots:
                if protected_robot["label"] != protected_class:
                    continue

                if not robot_touches_rotated_tower(protected_robot, tower):
                    continue

                for opponent_robot in robots:
                    if opponent_robot["tid"] == protected_robot["tid"]:
                        continue

                    if opponent_robot["label"] != opponent_class:
                        continue

                    if not robots_touch_for_foul(protected_robot, opponent_robot):
                        continue

                    contact_key = (
                        tower_alliance,
                        protected_robot["tid"],
                        opponent_robot["tid"]
                    )

                    if not contact_confirmed(contact_key):
                        continue

                    data = endgame_contact_memory[contact_key]
                    now = time.time()

                    if now - data.get("last_foul", 0) <= 1.0:
                        continue

                    data["last_foul"] = now

                    if tower_alliance == "BLUE":
                        red_endgame_fouls += 1
                        report_foul("RED", "MAJOR", "ENDGAME TOWER")
                    else:
                        blue_endgame_fouls += 1
                        report_foul("BLUE", "MAJOR", "ENDGAME TOWER")


    blue_count = blue_touch_count + sim_blue
    red_count = red_touch_count + sim_red


    if not override and match_phase == "PREMATCH" and blue_line_x is not None and red_line_x is not None:
        if blue_count == 3 and red_count == 3:
            ready_frames += 1
        else:
            ready_frames = 0


    if blue_line_x is not None:
        draw_projected_or_manual_line(annotated, "BLUE", blue_line_x, blue_line_angle, (255, 0, 0), h)
        cv2.putText(annotated, f"Blue: {blue_line_angle}°", (int(blue_line_x - 40), 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    if red_line_x is not None:
        draw_projected_or_manual_line(annotated, "RED", red_line_x, red_line_angle, (0, 0, 255), h)
        cv2.putText(annotated, f"Red: {red_line_angle}°", (int(red_line_x - 40), 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if center_line_x is not None:
        draw_projected_or_manual_line(annotated, "CENTER", center_line_x, center_line_angle, (0, 255, 0), h)
        cv2.putText(annotated, f"Center: {center_line_angle}°", (int(center_line_x - 50), 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


        overlay = annotated.copy()
        angle_rad = math.radians(center_line_angle)


        for y in range(0, h, 20):
            line_x_at_y = get_line_x_at_y(y, center_line_x, center_line_angle)

            if blue_on_right:

                cv2.line(overlay, (0, y), (int(line_x_at_y), y), (0, 0, 255), 1)

                cv2.line(overlay, (int(line_x_at_y), y), (w, y), (255, 0, 0), 1)
            else:

                cv2.line(overlay, (0, y), (int(line_x_at_y), y), (255, 0, 0), 1)

                cv2.line(overlay, (int(line_x_at_y), y), (w, y), (0, 0, 255), 1)

        cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)


        if blue_on_right:
            cv2.putText(annotated, "RED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.putText(annotated, "BLUE", (w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
        else:
            cv2.putText(annotated, "BLUE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
            cv2.putText(annotated, "RED", (w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


    draw_angled_line(annotated, preview_x, preview_angle, (170, 170, 170), 1, h)
    cv2.putText(
        annotated,
        f"Preview: {preview_angle}°",
        (max(5, preview_x - 60), 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (170, 170, 170),
        1
    )

    for tower_alliance, tower in [("BLUE", blue_tower), ("RED", red_tower)]:
        if tower is None:
            continue

        color = (255, 0, 0) if tower_alliance == "BLUE" else (0, 0, 255)

        tower_points = tower_ground_polygon(tower).astype(np.int32)
        selected = selected_field_object == f"{tower_alliance} TOWER"
        cv2.polylines(annotated, [tower_points], True,
                      (0, 255, 255) if selected else color,
                      3 if selected else 2)

        label_x = int(tower_points[:, 0].min())
        label_y = int(tower_points[:, 1].min()) - 8

        cv2.putText(
            annotated,
            f"{tower_alliance} TOWER BASE" if tower.get('auto') else f"{tower_alliance} TOWER",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            2
        )

    preview_alliance = get_alliance_at_point(
        preview_x,
        preview_y,
        center_line_x,
        center_line_angle,
        blue_on_right
    )

    preview_color = (255, 0, 0) if preview_alliance == "BLUE" else (0, 0, 255)

    preview_points = draw_rotated_rect(
        annotated,
        preview_x,
        preview_y,
        tower_width,
        tower_height,
        tower_preview_angle,
        (180, 180, 180),
        1
    )

    preview_label_x = int(preview_points[:, 0].min())
    preview_label_y = int(preview_points[:, 1].max()) + 15

    cv2.putText(
        annotated,
        f"{preview_alliance} TOWER PREVIEW {tower_preview_angle}°",
        (preview_label_x, preview_label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        preview_color,
        1
    )

    draw_field_and_pin_debug(annotated)
    if selected_field_object != 'NONE':
        cv2.putText(annotated, f"EDIT {selected_field_object}: A/D shift | I/K tower up/down | -/= rotate | L done",
                    (20, max(30, h - 158)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.49, (0, 255, 255), 2)

    if contact_debug:
        draw_contact_debug_overlay(annotated, robots, contact_debug_pairs)
        cv2.putText(
            annotated,
            f"EDGE OVERLAP >= {CONTACT_MIN_OVERLAP_RATIO * 100:.0f}% | GAP <= {CONTACT_GAP_PX}px | PIN CONF >= {PIN_CONFIDENCE_THRESHOLD:.2f}",
            (30, min(h - 132, 205)), cv2.FONT_HERSHEY_SIMPLEX,
            0.38, (255, 255, 255), 1,
        )


    if blue_line_x is None or red_line_x is None or center_line_x is None:
        match_phase = "WAITING_FOR_LINES"
        state = "WAITING FOR LINES"
        ready_frames = 0

    else:
        if match_phase == "WAITING_FOR_LINES":
            match_phase = "PREMATCH"

        if in_auto:
            state = "AUTO"

        elif in_waiting:
            state = "WAITING"

        elif in_transition:
            state = "TRANSITION"

        elif in_shifts:
            state = "SHIFTS"

        elif in_endgame:
            state = "ENDGAME"

        elif match_phase == "MATCH_OVER":
            state = "MATCH OVER"

        elif ready_frames >= 30:
            state = "MATCH READY"

        elif blue_count == 3 and red_count == 3:
            state = "PREMATCH - READY TO START"
        else:
            state = "PREMATCH"
            ready_frames = 0


    if phase_start_time is not None and match_phase in phase_durations:
        display_time = max(0.0, phase_time_remaining)
        timer_text = f"{match_phase}: {display_time:.1f}s"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2

        (text_w, text_h), baseline = cv2.getTextSize(
            timer_text,
            font,
            font_scale,
            thickness
        )

        timer_x = max(10, w - text_w - 20)
        timer_y = 35

        cv2.rectangle(
            annotated,
            (timer_x - 8, timer_y - text_h - 8),
            (w - 10, timer_y + baseline + 8),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            annotated,
            timer_text,
            (timer_x, timer_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

    y = 90

    for foul in penalties.blue_penalties:
        cv2.putText(
            annotated,
            f"BLUE: {foul}",
            (w - 350, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )
        y += 35

    for foul in penalties.red_penalties:
        cv2.putText(
            annotated,
            f"RED: {foul}",
            (w - 350, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        y += 35


    cv2.putText(annotated, state, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0) if "READY" in state else (255, 255, 0), 2)

    cv2.putText(annotated, f"BLUE TOUCH: {blue_count}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

    cv2.putText(annotated, f"RED TOUCH: {red_count}", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)


    draw_foul_announcements(annotated)

    if show_controls:
        controls_y = h - 125

        cv2.rectangle(
            annotated,
            (15, controls_y - 25),
            (w - 15, h - 15),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            annotated,
            "SPACE pause | LEFT/RIGHT 5s | HOME restart | ; calibrate+auto setup | TAB edit | A/D/I/K move | -/= rotate | L done",
            (30, controls_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (220, 220, 220),
            1
        )

        cv2.putText(
            annotated,
            "H debug | SHIFT+B blue major | SHIFT+R red major | B/R/C lines | S swap | Z next match | X reset | Q quit",
            (30, controls_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (220, 220, 220),
            1
        )

        cv2.putText(
            annotated,
            "F debug mode | Y blue/red | 5 add box | 6 delete | 7/8 select | A/D/I/K move selected debug box",
            (30, controls_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (220, 220, 220),
            1
        )
    else:
        cv2.putText(
            annotated,
            "V show controls",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1
        )

    if fullscreen_mode:
        display_frame = resize_to_fill_screen(annotated, screen_width, screen_height)
    else:
        display_frame = annotated

    cv2.imshow(WINDOW_NAME, display_frame)

    if not window_started:
        print("OpenCV window is running.")
        window_started = True


    key_raw = cv2.waitKeyEx(1)
    key = key_raw & 0xFF if key_raw != -1 else -1

    active_match = match_phase in {"AUTO", "WAITING", "TRANSITION", "SHIFTS", "ENDGAME"}
    if key_raw in (2424832, 65361) and hasattr(camera, "seek_relative"):
        if active_match:
            print("Seek blocked during match. Press Z or X to reset before reviewing.")
        else:
            success, seek_frame = camera.seek_relative(-5)
            if success:
                last_frame = seek_frame.copy()
                last_results = None
                reset_track_memory()
                print("BACK 5 SECONDS")

    elif key_raw in (2555904, 65363) and hasattr(camera, "seek_relative"):
        if active_match:
            print("Seek blocked during match. Press Z or X to reset before reviewing.")
        else:
            success, seek_frame = camera.seek_relative(5)
            if success:
                last_frame = seek_frame.copy()
                last_results = None
                reset_track_memory()
                print("FORWARD 5 SECONDS")

    elif key_raw in (2359296, 65360) and hasattr(camera, "restart"):
        if active_match:
            print("Restart blocked during match. Press Z or X to reset first.")
        else:
            success, seek_frame = camera.restart()
            if success:
                last_frame = seek_frame.copy()
                last_results = None
                reset_track_memory()
                print("VIDEO RESTART")

    elif key == ord(' '):
        paused = not paused

        if hasattr(camera, "set_paused"):
            camera.set_paused(paused)

        if paused:
            match_pause_start = time.time()
            print("PAUSED")
        else:
            if match_pause_start is not None:
                pause_duration = time.time() - match_pause_start

                if phase_start_time is not None:
                    phase_start_time += pause_duration


                for pin in active_pins.values():
                    pin["last_update"] += pause_duration
                    if pin.get("a_since") is not None:
                        pin["a_since"] += pause_duration
                    if pin.get("b_since") is not None:
                        pin["b_since"] += pause_duration
                    if pin.get("missing_since") is not None:
                        pin["missing_since"] += pause_duration
                    if pin.get("uncertain_since") is not None:
                        pin["uncertain_since"] += pause_duration

                for candidate in pin_candidate_memory.values():
                    candidate["since"] += pause_duration
                    candidate["last_seen"] += pause_duration

                for tid, history in robot_field_history.items():
                    robot_field_history[tid] = [
                        (sample_time + pause_duration, x, y)
                        for sample_time, x, y in history
                    ]

            match_pause_start = None
            print("PLAYING")

    elif key == 9:
        if field_homography is not None and field_auto_setup_enabled:
            current = FIELD_EDIT_OBJECTS.index(selected_field_object)
            selected_field_object = FIELD_EDIT_OBJECTS[(current + 1) % len(FIELD_EDIT_OBJECTS)]
            print(f"FIELD EDIT: {selected_field_object}")
        else:
            print("Calibrate the four field corners first (;).")

    elif key == ord('a'):
        move_step = get_move_step()
        robot = selected_debug_robot()
        if selected_field_object != 'NONE' and not debug_mode:
            adjust_selected_field_element(dx=-move_step)
        elif debug_mode and robot is not None:
            move_debug_robot(robot, -move_step, 0)
        else:
            preview_x -= move_step

    elif key == ord('d'):
        move_step = get_move_step()
        robot = selected_debug_robot()
        if selected_field_object != 'NONE' and not debug_mode:
            adjust_selected_field_element(dx=move_step)
        elif debug_mode and robot is not None:
            move_debug_robot(robot, move_step, 0)
        else:
            preview_x += move_step

    elif key == ord('-'):
        angle_step = get_angle_step()

        if selected_field_object != 'NONE' and not debug_mode:
            adjust_selected_field_element(angle_delta=-angle_step)
        elif stage == "BLUE" and blue_line_x is not None:
            blue_line_angle -= angle_step
        elif stage == "RED" and red_line_x is not None:
            red_line_angle -= angle_step
        elif stage == "CENTER" and center_line_x is not None:
            center_line_angle -= angle_step
        else:
            preview_angle -= angle_step

    elif key == ord('='):
        angle_step = get_angle_step()

        if selected_field_object != 'NONE' and not debug_mode:
            adjust_selected_field_element(angle_delta=angle_step)
        elif stage == "BLUE" and blue_line_x is not None:
            blue_line_angle += angle_step
        elif stage == "RED" and red_line_x is not None:
            red_line_angle += angle_step
        elif stage == "CENTER" and center_line_x is not None:
            center_line_angle += angle_step
        else:
            preview_angle += angle_step

    elif key == ord('i'):
        move_step = get_move_step()
        robot = selected_debug_robot()
        if selected_field_object != 'NONE' and not debug_mode:
            adjust_selected_field_element(dy=-move_step)
        elif debug_mode and robot is not None:
            move_debug_robot(robot, 0, -move_step)
        else:
            preview_y = max(0, preview_y - move_step)

    elif key == ord('k'):
        move_step = get_move_step()
        robot = selected_debug_robot()
        if selected_field_object != 'NONE' and not debug_mode:
            adjust_selected_field_element(dy=move_step)
        elif debug_mode and robot is not None:
            move_debug_robot(robot, 0, move_step)
        else:
            preview_y = min(h, preview_y + move_step)

    elif key == ord(';'):
        if fullscreen_mode:
            print("Turn fullscreen OFF before field calibration.")
        else:
            field_calibration_points.clear()
            field_homography = None
            field_calibration_mode = True
            selected_field_object = 'NONE'
            auto_line_segments.clear()
            field_auto_setup_enabled = False
            print("FIELD CALIBRATION START")
            print("Click 4 field corners in order: 1 -> 2 along the LONG side, then 3, then 4.")

    elif key == ord('\\'):
        if not apply_calibrated_field_setup():
            print('AUTO SETUP FAILED: redo the four field corners with ;')

    elif key == ord('h') or key == ord('H'):
        contact_debug = not contact_debug

        if contact_debug:
            print("CONTACT DEBUG ON")
        else:
            print("CONTACT DEBUG OFF")

    elif key == ord('v') or key == ord('V'):
        show_controls = not show_controls

        if show_controls:
            print("CONTROLS SHOWN")
        else:
            print("CONTROLS HIDDEN")


    elif key == ord('9'):
        fullscreen_mode = not fullscreen_mode

        if fullscreen_mode:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print(f"FULLSCREEN ON: {screen_width}x{screen_height}")
        else:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
            print("FULLSCREEN OFF")

    elif key == ord('.'):
        precision_mode = (precision_mode + 1) % 3

        if precision_mode == 0:
            print("PRECISION MODE: STANDARD")
        elif precision_mode == 1:
            print("PRECISION MODE: PRECISE")
        else:
            print("PRECISION MODE: SUPER PRECISE")

    elif key == ord('f') or key == ord('F'):
        debug_mode = not debug_mode

        if debug_mode:
            print("DEBUG ROBOT MODE ON")
        else:
            print("DEBUG ROBOT MODE OFF")

    elif key == ord('y') or key == ord('Y'):
        if debug_robot_label == BLUE_CLASS:
            debug_robot_label = RED_CLASS
            print("DEBUG ROBOT ALLIANCE: RED")
        else:
            debug_robot_label = BLUE_CLASS
            print("DEBUG ROBOT ALLIANCE: BLUE")

    elif key == ord('5'):
        debug_robots.append(create_debug_robot(preview_x, preview_y, debug_robot_label))
        selected_debug_robot_index = len(debug_robots) - 1
        print("DEBUG ROBOT ADDED")

    elif key == ord('6'):
        if selected_debug_robot_index is not None and len(debug_robots) > 0:
            debug_robots.pop(selected_debug_robot_index)

            if len(debug_robots) == 0:
                selected_debug_robot_index = None
            else:
                selected_debug_robot_index = min(selected_debug_robot_index, len(debug_robots) - 1)

            print("DEBUG ROBOT DELETED")

    elif key == ord('7'):
        if len(debug_robots) > 0:
            if selected_debug_robot_index is None:
                selected_debug_robot_index = 0
            else:
                selected_debug_robot_index = (selected_debug_robot_index - 1) % len(debug_robots)

            print(f"SELECTED DEBUG ROBOT {selected_debug_robot_index + 1}/{len(debug_robots)}")

    elif key == ord('8'):
        if len(debug_robots) > 0:
            if selected_debug_robot_index is None:
                selected_debug_robot_index = 0
            else:
                selected_debug_robot_index = (selected_debug_robot_index + 1) % len(debug_robots)

            print(f"SELECTED DEBUG ROBOT {selected_debug_robot_index + 1}/{len(debug_robots)}")

    elif key == ord('e') or key == ord('E'):
        if blue_line_x is not None and red_line_x is not None and center_line_x is not None:
            match_phase = "ENDGAME"
            phase_start_time = time.time()

            if paused:
                match_pause_start = phase_start_time

            print("ENDGAME TEST START")
        else:
            print("Set blue, red, and center lines before endgame test.")

    elif key == ord('t'):
        selected_field_object = 'NONE'
        tower_alliance = get_alliance_at_point(
            preview_x,
            preview_y,
            center_line_x,
            center_line_angle,
            blue_on_right
        )

        if tower_alliance == "BLUE":
            blue_tower = {
                "x": preview_x,
                "y": preview_y,
                "angle": tower_preview_angle
            }
            stage = "BLUE_TOWER"
            print(f"BLUE tower placed at ({preview_x}, {preview_y}), angle={tower_preview_angle}°")
        else:
            red_tower = {
                "x": preview_x,
                "y": preview_y,
                "angle": tower_preview_angle
            }
            stage = "RED_TOWER"
            print(f"RED tower placed at ({preview_x}, {preview_y}), angle={tower_preview_angle}°")

    elif key == ord('g'):
        tower_alliance = get_alliance_at_point(
            preview_x,
            preview_y,
            center_line_x,
            center_line_angle,
            blue_on_right
        )

        if tower_alliance == "BLUE":
            blue_tower = None
            print("BLUE tower cleared")
        else:
            red_tower = None
            print("RED tower cleared")

    elif key == ord('u'):
        size_step = get_size_step()
        robot = selected_debug_robot()

        if selected_field_object in ('BLUE TOWER', 'RED TOWER') and not debug_mode:
            adjust_selected_field_element(scale_y=1.05)
        elif debug_mode:
            resize_debug_robot(robot, 0, size_step)
        else:
            tower_height += size_step
            print(f"Tower height: {tower_height}")

    elif key == ord('j'):
        size_step = get_size_step()
        robot = selected_debug_robot()

        if selected_field_object in ('BLUE TOWER', 'RED TOWER') and not debug_mode:
            adjust_selected_field_element(scale_y=0.95)
        elif debug_mode:
            resize_debug_robot(robot, 0, -size_step)
        else:
            tower_height = max(20, tower_height - size_step)
            print(f"Tower height: {tower_height}")

    elif key == ord('['):
        size_step = get_size_step()
        robot = selected_debug_robot()

        if selected_field_object in ('BLUE TOWER', 'RED TOWER') and not debug_mode:
            adjust_selected_field_element(scale_x=0.95)
        elif debug_mode:
            resize_debug_robot(robot, -size_step, 0)
        else:
            tower_width = max(20, tower_width - size_step)
            print(f"Tower width: {tower_width}")

    elif key == ord(']'):
        size_step = get_size_step()
        robot = selected_debug_robot()

        if selected_field_object in ('BLUE TOWER', 'RED TOWER') and not debug_mode:
            adjust_selected_field_element(scale_x=1.05)
        elif debug_mode:
            resize_debug_robot(robot, size_step, 0)
        else:
            tower_width += size_step
            print(f"Tower width: {tower_width}")

    elif key == ord('n'):
        tower_preview_angle -= get_angle_step()
        print(f"Tower preview angle: {tower_preview_angle}°")

    elif key == ord('m'):
        tower_preview_angle += get_angle_step()
        print(f"Tower preview angle: {tower_preview_angle}°")

    elif key == ord('B'):
        report_foul("BLUE", "MAJOR", "MANUAL KEYBOARD")

    elif key == ord('R'):
        report_foul("RED", "MAJOR", "MANUAL KEYBOARD")

    elif key == ord('b'):
        auto_line_segments.pop('BLUE', None)
        selected_field_object = 'NONE'
        blue_line_x = preview_x
        blue_line_angle = preview_angle
        stage = "BLUE"
        print(f"Blue line set at x={blue_line_x}, angle={blue_line_angle}°")

    elif key == ord('r'):
        auto_line_segments.pop('RED', None)
        selected_field_object = 'NONE'
        red_line_x = preview_x
        red_line_angle = preview_angle
        stage = "RED"
        print(f"Red line set at x={red_line_x}, angle={red_line_angle}°")

    elif key == ord('c'):
        auto_line_segments.pop('CENTER', None)
        selected_field_object = 'NONE'
        center_line_x = preview_x
        center_line_angle = preview_angle
        stage = "CENTER"
        print(f"Center line set at x={center_line_x}, angle={center_line_angle}°")

    elif key == ord('l'):
        if selected_field_object != 'NONE':
            print(f"FINISHED EDITING: {selected_field_object}")
            selected_field_object = 'NONE'
        if stage == "BLUE" and blue_line_x is not None:
            print(f"Blue line locked at angle {blue_line_angle}°")
        elif stage == "RED" and red_line_x is not None:
            print(f"Red line locked at angle {red_line_angle}°")
        elif stage == "CENTER" and center_line_x is not None:
            print(f"Center line locked at angle {center_line_angle}°")
        elif stage == "BLUE_TOWER" and blue_tower is not None:
            print(f"Blue tower locked at ({blue_tower['x']}, {blue_tower['y']}), angle={blue_tower['angle']}°")
        elif stage == "RED_TOWER" and red_tower is not None:
            print(f"Red tower locked at ({red_tower['x']}, {red_tower['y']}), angle={red_tower['angle']}°")
        else:
            print("Preview mode active")

        stage = "PREVIEW"

    elif key == ord('s'):

        blue_on_right = not blue_on_right
        if field_auto_setup_enabled:
            swap_calibrated_alliance_objects()
        if blue_on_right:
            print("BLUE on right, RED on left")
        else:
            print("BLUE on left, RED on right")
    elif key == ord('z') or key == ord('Z'):
        if foul_log:
            print_foul_report("SOFT RESET FOUL REPORT")

        if blue_line_x is not None and red_line_x is not None and center_line_x is not None:
            match_phase = "PREMATCH"
            state = "PREMATCH"
        else:
            match_phase = "WAITING_FOR_LINES"
            state = "WAITING FOR LINES"

        phase_start_time = None
        ready_frames = 0
        override = False
        match_pause_start = None

        sim_blue = 0
        sim_red = 0

        blue_endgame_fouls = 0
        red_endgame_fouls = 0

        endgame_contact_memory.clear()
        auto_contact_memory.clear()
        pin_candidate_memory.clear()
        active_pins.clear()
        robot_field_history.clear()
        reset_track_memory()
        debug_robots.clear()
        selected_debug_robot_index = None
        penalties.reset()
        contact_debug_pairs.clear()

        foul_log.clear()
        foul_announcements.clear()

        foul_totals["BLUE"]["MAJOR"] = 0
        foul_totals["BLUE"]["MINOR"] = 0
        foul_totals["RED"]["MAJOR"] = 0
        foul_totals["RED"]["MINOR"] = 0

        match_reported = False

        print("Soft reset for next match. Lines and towers kept.")

    elif key == ord('x'):
        selected_field_object = 'NONE'
        field_auto_setup_enabled = False
        auto_line_segments.clear()
        blue_line_x = None
        red_line_x = None
        center_line_x = None
        blue_line_angle = 0
        red_line_angle = 0
        center_line_angle = 0
        preview_angle = 0
        tower_preview_angle = 0
        match_phase = "WAITING_FOR_LINES"
        phase_start_time = None
        ready_frames = 0
        override = False
        blue_tower = None
        red_tower = None
        endgame_contact_memory.clear()
        auto_contact_memory.clear()
        pin_candidate_memory.clear()
        active_pins.clear()
        robot_field_history.clear()
        reset_track_memory()
        if foul_log:
            print_foul_report("MATCH RESET FOUL REPORT")

        debug_robots.clear()
        selected_debug_robot_index = None
        penalties.reset()
        contact_debug_pairs.clear()

        foul_log.clear()
        foul_announcements.clear()

        foul_totals["BLUE"]["MAJOR"] = 0
        foul_totals["BLUE"]["MINOR"] = 0
        foul_totals["RED"]["MAJOR"] = 0
        foul_totals["RED"]["MINOR"] = 0

        match_reported = False

        print("Lines reset")

    elif key == ord('p') or key == ord('P'):
        if blue_line_x is not None and red_line_x is not None and center_line_x is not None:
            match_phase = "AUTO"
            phase_start_time = time.time()
            ready_frames = 0
            override = False
            auto_contact_memory.clear()
            pin_candidate_memory.clear()
            active_pins.clear()
            robot_field_history.clear()

            if paused:
                match_pause_start = phase_start_time

            print("MATCH START")
        else:
            print("Set blue, red, and center lines before starting.")

    elif key == ord('o') or key == ord('O') or key == ord('0'):
        if blue_line_x is not None and red_line_x is not None and center_line_x is not None:
            match_phase = "AUTO"
            phase_start_time = time.time()
            ready_frames = 0
            override = False
            auto_contact_memory.clear()
            pin_candidate_memory.clear()
            active_pins.clear()
            robot_field_history.clear()

            if paused:
                match_pause_start = phase_start_time

            print("OVERRIDE - MATCH START")
        else:
            print("Set blue, red, and center lines before starting.")

    elif key == ord('1'):
        sim_blue += 1
    elif key == ord('2'):
        sim_blue = max(0, sim_blue - 1)

    elif key == ord('3'):
        sim_red += 1
    elif key == ord('4'):
        sim_red = max(0, sim_red - 1)

    elif key == ord('q'):
        break


yolo_worker_running = False
camera.release()
cv2.destroyAllWindows()
