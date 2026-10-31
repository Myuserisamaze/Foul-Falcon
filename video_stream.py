import os
import glob
import time

import cv2
import yt_dlp


class VideoStream:
    def __init__(self, url, target_width=1280):
        self.url = url
        self.target_width = target_width
        self.paused = False
        self.latest_frame = None

        project_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(project_dir, "video_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        cookie_path = os.path.join(project_dir, "youtube_cookies.txt")

        deno_matches = glob.glob(
            os.path.join(
                os.environ.get("LOCALAPPDATA", ""),
                "Microsoft",
                "WinGet",
                "Packages",
                "DenoLand.Deno_*",
                "deno.exe",
            )
        )

        self.ydl_opts = {
            "format": (
                "bestvideo[height<=1080][ext=mp4]/"
                "bestvideo[height<=1080]/"
                "best[height<=1080][ext=mp4]/"
                "best[height<=1080]"
            ),
            "outtmpl": os.path.join(
                self.cache_dir,
                "%(id)s.%(ext)s"
            ),
            "noplaylist": True,
            "continuedl": True,
            "overwrites": False,
            "quiet": False,
        }

        if os.path.exists(cookie_path):
            self.ydl_opts["cookiefile"] = cookie_path

        if deno_matches:
            self.ydl_opts["remote_components"] = {"ejs:github"}
            self.ydl_opts["js_runtimes"] = {
                "deno": {
                    "path": deno_matches[0]
                }
            }

        self.local_path = self._download_video()

        self.cap = cv2.VideoCapture(self.local_path)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open cached video: {self.local_path}"
            )

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.fps is None or self.fps <= 0 or self.fps > 120:
            self.fps = 30.0

        self.frame_delay = 1.0 / self.fps
        self.next_playback_time = time.perf_counter()

        frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if frame_count > 0:
            self.duration_seconds = frame_count / self.fps
        else:
            self.duration_seconds = 0.0

        print(f"LOCAL VIDEO: {self.local_path}")
        print(f"VIDEO FPS: {self.fps:.2f}")

        if self.duration_seconds > 0:
            print(
                f"VIDEO LENGTH: "
                f"{self._format_time(self.duration_seconds)}"
            )

    def _download_video(self):
        print("Preparing YouTube video...")

        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            info = ydl.extract_info(
                self.url,
                download=True
            )

            candidate_paths = []

            requested_downloads = (
                info.get("requested_downloads") or []
            )

            for item in requested_downloads:
                filepath = item.get("filepath")

                if filepath:
                    candidate_paths.append(filepath)

            filename = info.get("_filename")

            if filename:
                candidate_paths.append(filename)

            try:
                candidate_paths.append(
                    ydl.prepare_filename(info)
                )
            except Exception:
                pass

            video_id = info.get("id")

            if video_id:
                candidate_paths.extend(
                    glob.glob(
                        os.path.join(
                            self.cache_dir,
                            f"{video_id}.*"
                        )
                    )
                )

        for path in candidate_paths:
            if not path:
                continue

            lower = path.lower()

            if lower.endswith(
                (".part", ".ytdl", ".json")
            ):
                continue

            if os.path.isfile(path):
                return os.path.abspath(path)

        raise FileNotFoundError(
            "yt-dlp finished, but the downloaded "
            "video file could not be found."
        )

    @staticmethod
    def _format_time(seconds):
        seconds = max(0, int(seconds))

        minutes, seconds = divmod(
            seconds,
            60
        )

        hours, minutes = divmod(
            minutes,
            60
        )

        if hours:
            return (
                f"{hours}:"
                f"{minutes:02d}:"
                f"{seconds:02d}"
            )

        return f"{minutes}:{seconds:02d}"

    def _resize_frame(self, frame):
        height, width = frame.shape[:2]

        if width <= self.target_width:
            return frame

        scale = self.target_width / width

        new_width = int(width * scale)
        new_height = int(height * scale)

        return cv2.resize(
            frame,
            (new_width, new_height)
        )

    def _read_frame_without_pacing(self):
        success, frame = self.cap.read()

        if not success:
            return False, None

        frame = self._resize_frame(frame)

        self.latest_frame = frame.copy()

        self.next_playback_time = (
            time.perf_counter()
            + self.frame_delay
        )

        return True, frame

    def read(self):
        if self.paused and self.latest_frame is not None:
            return True, self.latest_frame.copy()

        success, frame = self.cap.read()

        if not success:
            return False, None

        frame = self._resize_frame(frame)

        now = time.perf_counter()

        sleep_time = (
            self.next_playback_time - now
        )

        if sleep_time > 0:
            time.sleep(sleep_time)

        self.next_playback_time += self.frame_delay

        if (
            self.next_playback_time
            < time.perf_counter()
            - self.frame_delay
        ):
            self.next_playback_time = (
                time.perf_counter()
            )

        self.latest_frame = frame.copy()

        return True, frame

    def get_position_seconds(self):
        position_ms = self.cap.get(
            cv2.CAP_PROP_POS_MSEC
        )

        if position_ms is None or position_ms < 0:
            return 0.0

        return position_ms / 1000.0

    def seek_to_seconds(self, seconds):
        target = max(
            0.0,
            float(seconds)
        )

        if self.duration_seconds > 0:
            target = min(
                target,
                max(
                    0.0,
                    self.duration_seconds
                    - self.frame_delay
                )
            )

        self.cap.set(
            cv2.CAP_PROP_POS_MSEC,
            target * 1000.0
        )

        success, frame = (
            self._read_frame_without_pacing()
        )

        if success:
            actual = self.get_position_seconds()

            print(
                f"VIDEO SEEK: "
                f"{self._format_time(actual)}"
            )

        return success, frame

    def seek_relative(self, seconds):
        return self.seek_to_seconds(
            self.get_position_seconds()
            + float(seconds)
        )

    def restart(self):
        print("VIDEO RESTART")

        return self.seek_to_seconds(0.0)

    def set_paused(self, paused):
        self.paused = paused

        self.next_playback_time = (
            time.perf_counter()
        )

    def release(self):
        if self.cap is not None:
            self.cap.release()