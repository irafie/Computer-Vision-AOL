import cv2
import numpy as np
from collections import deque

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

ORB_N_FEATURES = 250
MATCH_RATIO = 0.7

WINDOW_SIZE = 30
MIN_KEYPOINTS = 15

SAFE_DIST_RATIO = 0.32
HEAD_TILT_THRESH = 0.24
BODY_LEAN_THRESH = 0.22
SHOULDER_BALANCE_THRESH = 0.40

GREEN = (0, 255, 100)
RED = (50, 50, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 180, 50)
WHITE = (240, 240, 240)
DARK = (20, 20, 20)


def preprocess_plain(frame):

    return cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Pakai noise
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # noise = np.random.normal(0, 20, gray.shape).astype(np.int16)
    # noisy = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # return noisy


def preprocess_filtered(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    bilateral = cv2.bilateralFilter(gray, 5, 50, 50)
    
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    return clahe.apply(bilateral)

    # Pakai noise
    # noise = np.random.normal(0, 20, gray.shape).astype(np.int16)
    # noisy = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # bilateral = cv2.bilateralFilter(noisy, 5, 50, 50)
    # clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    
    # return clahe.apply(bilateral)


def orb_detect_describe(gray_roi):

    orb = cv2.ORB_create(
        nfeatures=ORB_N_FEATURES,
        scoreType=cv2.ORB_HARRIS_SCORE
    )

    keypoints, descriptors = orb.detectAndCompute(
        gray_roi,
        None
    )

    return keypoints, descriptors


def match_descriptors(des1, des2):

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except:
        return 0

    good_matches = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < MATCH_RATIO * n.distance:
            good_matches.append(m)

    return len(good_matches)


face_cascade = cv2.CascadeClassifier(
    FACE_CASCADE_PATH
)


def detect_face(gray):

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    return max(
        faces,
        key=lambda f: f[2] * f[3]
    )


def upper_body_roi(face, frame_shape):

    fh, fw = frame_shape[:2]

    x, y, w, h = face

    margin_x = int(w * 0.55)

    top_margin = int(h * 0.2)

    bottom_margin = int(h * 1.5)

    rx = max(0, x - margin_x)
    ry = max(0, y - top_margin)

    rx2 = min(fw, x + w + margin_x)
    ry2 = min(fh, y + h + bottom_margin)

    return rx, ry, rx2, ry2


def analyse_posture(face, keypoints, roi_offset, frame_shape):

    fh, fw = frame_shape[:2]

    x, y, w, h = face

    warnings = []

    if w / fw > SAFE_DIST_RATIO:
        warnings.append("Terlalu dekat layar")

    face_center_x = (x + w / 2) / fw

    if abs(face_center_x - 0.5) > HEAD_TILT_THRESH:

        if face_center_x > 0.5:
            warnings.append("Kepala miring kanan")
        else:
            warnings.append("Kepala miring kiri")

    if len(keypoints) >= MIN_KEYPOINTS:

        rx, ry = roi_offset

        pts = np.array([
            [kp.pt[0] + rx, kp.pt[1] + ry]
            for kp in keypoints
        ])
        

        centroid_x = pts[:, 0].mean() / fw

        if abs(centroid_x - 0.5) > BODY_LEAN_THRESH:

            if centroid_x > 0.5:
                warnings.append("Badan condong kanan")
            else:
                warnings.append("Badan condong kiri")

        body_center = pts[:, 0].mean()

        left_points = pts[pts[:, 0] < body_center]
        right_points = pts[pts[:, 0] >= body_center]

        if len(left_points) > 0 and len(right_points) > 0:

            imbalance = abs(
                len(left_points) - len(right_points)
            ) / max(
                len(left_points),
                len(right_points)
            )

            if imbalance > SHOULDER_BALANCE_THRESH:
                warnings.append("Bahu tidak seimbang")

    if len(warnings) == 0:
        label = "POSTUR BAIK"
    else:
        label = "POSTUR BURUK"

    return label, warnings


def put(img, text, pos,
        color=WHITE,
        scale=0.5,
        thick=1):

    cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        DARK,
        thick + 2,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thick,
        cv2.LINE_AA
    )


def draw_keypoints(img, keypoints, roi_offset):

    rx, ry = roi_offset

    for kp in keypoints:

        px = int(kp.pt[0]) + rx
        py = int(kp.pt[1]) + ry

        cv2.circle(
            img,
            (px, py),
            3,
            YELLOW,
            -1
        )


def draw_warnings(img, warnings):

    h = img.shape[0]

    start_y = h - 20 - len(warnings) * 28

    for i, warning in enumerate(warnings):

        put(
            img,
            f"! {warning}",
            (10, start_y + i * 28),
            RED,
            0.58,
            1
        )


class AccuracyTracker:

    def __init__(self, window=WINDOW_SIZE):

        self.history = deque(maxlen=window)

    def update(self, value):

        self.history.append(value)

    @property
    def rate(self):

        if len(self.history) == 0:
            return 0

        return 100 * sum(self.history) / len(self.history)


def process_pipeline(
        frame,
        gray,
        previous_descriptors,
        tracker,
        title
):

    vis = frame.copy()

    fh, fw = frame.shape[:2]

    tracking_score = 0

    face = detect_face(gray)

    if face is not None:

        x, y, w, h = face

        rx, ry, rx2, ry2 = upper_body_roi(
            face,
            frame.shape
        )

        roi = gray[ry:ry2, rx:rx2]

        

        edges = cv2.Canny(
            roi,
            80,
            150
        )

        edge_count = np.sum(edges > 0)

        keypoints, descriptors = orb_detect_describe(
            roi
        )

        good_matches = 0

        if previous_descriptors is not None and descriptors is not None:

            good_matches = match_descriptors(
                previous_descriptors,
                descriptors
            )

            total_features = max(
                len(keypoints),
                1
            )

            tracking_score = min(
                good_matches / (total_features * 0.7),
                1.0
            )

        elif descriptors is not None:

            tracking_score = 0.5

        posture, warnings = analyse_posture(
            face,
            keypoints,
            (rx, ry),
            frame.shape
        )

        color = GREEN if posture == "POSTUR BAIK" else RED

        cv2.rectangle(
            vis,
            (rx, ry),
            (rx2, ry2),
            BLUE,
            2
        )

        cv2.rectangle(
            vis,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.line(
            vis,
            (fw // 2, 0),
            (fw // 2, fh),
            WHITE,
            1
        )

        draw_keypoints(
            vis,
            keypoints,
            (rx, ry)
        )

        put(
            vis,
            posture,
            (x, y - 10),
            color,
            0.65,
            2
        )

        put(
            vis,
            f"ORB Keypoints: {len(keypoints)}",
            (10, 55),
            YELLOW,
            0.55,
            1
        )

        put(
            vis,
            f"ORB Match: {good_matches}",
            (10, 80),
            YELLOW,
            0.55,
            1
        )

        put(
            vis,
            f"Tracking Score: {tracking_score*100:.1f}%",
            (10, 105),
            GREEN,
            0.55,
            1
        )

        put(
            vis,
            f"Edges: {edge_count}",
            (10, 130),
            BLUE,
            0.55,
            1
        )

        if len(warnings) > 0:
            draw_warnings(vis, warnings)

    else:

        descriptors = None

        put(
            vis,
            "Wajah tidak terdeteksi",
            (20, fh // 2),
            RED,
            0.8,
            2
        )

    tracker.update(tracking_score)

    accuracy = tracker.rate

    put(
        vis,
        title,
        (10, 25),
        WHITE,
        0.7,
        2
    )

    put(
        vis,
        f"Akurasi: {accuracy:.1f}%",
        (10, 160),
        GREEN if accuracy >= 70 else RED,
        0.65,
        2
    )

    return vis, descriptors


def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("Webcam tidak ditemukan")
        return

    tracker_plain = AccuracyTracker()
    tracker_filter = AccuracyTracker()

    prev_des_plain = None
    prev_des_filter = None

    print("SYSTEM STARTED")
    print("Tekan Q untuk keluar")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        gray_plain = preprocess_plain(frame)

        gray_filter = preprocess_filtered(frame)

        vis_plain, prev_des_plain = process_pipeline(
            frame,
            gray_plain,
            prev_des_plain,
            tracker_plain,
            "TANPA FILTER"
        )

        vis_filter, prev_des_filter = process_pipeline(
            frame,
            gray_filter,
            prev_des_filter,
            tracker_filter,
            "DENGAN FILTER"
        )

        target_w = 640

        fh, fw = frame.shape[:2]

        ratio = target_w / fw

        new_h = int(fh * ratio)

        vis_plain = cv2.resize(
            vis_plain,
            (target_w, new_h)
        )

        vis_filter = cv2.resize(
            vis_filter,
            (target_w, new_h)
        )

        combined = np.hstack([
            vis_plain,
            vis_filter
        ])

        panel = np.zeros(
            (90, combined.shape[1], 3),
            dtype=np.uint8
        )

        panel[:] = (25, 25, 25)

        plain_acc = tracker_plain.rate
        filter_acc = tracker_filter.rate

        diff = filter_acc - plain_acc

        put(
            panel,
            f"Tanpa Filter : {plain_acc:.1f}%",
            (20, 30),
            WHITE,
            0.6,
            2
        )

        put(
            panel,
            f"Dengan Filter : {filter_acc:.1f}%",
            (430, 30),
            WHITE,
            0.6,
            2
        )

        if diff >= 0:

            txt = f"Filter lebih baik : {diff:.1f}%"
            color = GREEN

        else:

            txt = f"Filter lebih buruk : {abs(diff):.1f}%"
            color = RED

        put(
            panel,
            txt,
            (20, 65),
            color,
            0.6,
            2
        )

        put(
            panel,
            "Tekan Q untuk keluar",
            (950, 65),
            WHITE,
            0.5,
            1
        )

        output = np.vstack([
            combined,
            panel
        ])

        cv2.imshow(
            "POSTURE MONITOR - ORB",
            output
        )

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
