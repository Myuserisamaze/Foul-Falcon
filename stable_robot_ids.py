"""Persistent tracking for FRC footage, without suppressing raw YOLO detections.

The first three track slots for each alliance are reserved across the match.
Additional detections receive short-lived auxiliary track IDs instead of None.
Call update() once per NEW inference result and reuse the returned IDs on
intermediate display frames. IDs are NOT actual FRC team numbers.
"""
from __future__ import annotations

import itertools
import math


def _iou(a, b):
    x1 = max(float(a['x1']), float(b['x1']))
    y1 = max(float(a['y1']), float(b['y1']))
    x2 = min(float(a['x2']), float(b['x2']))
    y2 = min(float(a['y2']), float(b['y2']))
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, float(a['x2']) - a['x1']) * max(0, float(a['y2']) - a['y1'])
    bb = max(0, float(b['x2']) - b['x1']) * max(0, float(b['y2']) - b['y1'])
    return inter / max(1.0, aa + bb - inter)


class StableRobotTracker:
    def __init__(self, class_limits):
        self.class_limits = dict(class_limits)
        self.reset()

    def reset(self):
        self.tracks = {}
        self.next_id = 0
        self.last_frame = -1

    @staticmethod
    def _prediction(track, frame):
        age = max(0, frame - track['last_frame'])
        if age > 45:
            return track['cx'], track['cy']
        horizon = min(age, 16)
        return (track['cx'] + track['vx'] * horizon,
                track['cy'] + track['vy'] * horizon)

    @staticmethod
    def _match_cost(track, det, frame, force=False):
        age = max(0, frame - track['last_frame'])
        px, py = StableRobotTracker._prediction(track, frame)
        dist = math.hypot(float(det['cx']) - px, float(det['cy']) - py)
        width = max(1, float(det['x2']) - det['x1'], track['w'])
        height = max(1, float(det['y2']) - det['y1'], track['h'])
        gate = max(85, 1.6 * math.hypot(width, height)) + min(430, age * 3.0)
        if dist > gate and not force:
            return None
        spatial = dist / max(gate, 1)
        # IoU only helps when observations are nearby in time.
        overlap = _iou(track['box'], det) if age < 20 else 0.0
        new_ratio = max(0.1, (det['x2'] - det['x1']) * (det['y2'] - det['y1']))
        old_ratio = max(0.1, track['w'] * track['h'])
        size_mismatch = min(2.0, abs(math.log(new_ratio / old_ratio)))
        return spatial + 0.14 * (1.0 - overlap) + 0.09 * size_mismatch

    def _record_observation(self, tid, det, frame):
        data = self.tracks[tid]
        elapsed = max(1, frame - data['last_frame'])
        measured_vx = (float(det['cx']) - data['cx']) / elapsed
        measured_vy = (float(det['cy']) - data['cy']) / elapsed
        # Damp detector jitter; constrain implausible velocities.
        data['vx'] = max(-20, min(20, 0.45 * measured_vx + 0.55 * data['vx']))
        data['vy'] = max(-20, min(20, 0.45 * measured_vy + 0.55 * data['vy']))
        data['cx'], data['cy'] = float(det['cx']), float(det['cy'])
        data['w'], data['h'] = float(det['x2'] - det['x1']), float(det['y2'] - det['y1'])
        data['box'] = dict(det)
        data['last_frame'] = frame
        data['hits'] += 1

    def update(self, detections, frame_index):
        """Return one integer track ID per input detection, never None.

        Three core tracks per alliance are retained indefinitely to tolerate
        missed detections. Overflow tracks are temporary and do not consume
        core slots. Every YOLO box remains visible and available downstream.
        """
        frame_index = int(frame_index)
        if frame_index <= self.last_frame:
            self.reset()
        self.last_frame = frame_index
        ids = [None] * len(detections)

        # Retire only auxiliary tracks after approximately 2 seconds of absence
        # at 60 FPS. Permanent alliance tracks must survive occlusion.
        for tid, data in list(self.tracks.items()):
            if data.get('overflow') and frame_index - data['last_frame'] > 120:
                del self.tracks[tid]

        for label, limit in self.class_limits.items():
            all_indices = [i for i, det in enumerate(detections) if det['label'] == label]
            if not all_indices:
                continue
            tracks = [(tid, data) for tid, data in self.tracks.items()
                      if data['label'] == label and not data.get('overflow')]
            # Consider the eight strongest candidates for the three permanent
            # identities. All remaining detections still receive an auxiliary ID.
            indices = list(all_indices)
            indices.sort(key=lambda i: -float(detections[i].get('confidence', 0.0)))
            indices = indices[:8]
            filtered = []
            for i in indices:
                if all(_iou(detections[i], detections[j]) < 0.86 for j in filtered):
                    filtered.append(i)
            indices = filtered

            # Brute force the globally best assignment for up to three old tracks.
            # This avoids the greedy swap where the first processed box steals an ID.
            best_score = 0.0
            best_matches = []
            if tracks:
                def search(k, taken, score, matches):
                    nonlocal best_score, best_matches
                    if k >= len(tracks):
                        if score > best_score:
                            best_score, best_matches = score, list(matches)
                        return
                    tid, data = tracks[k]
                    search(k + 1, taken, score, matches)
                    for i in indices:
                        if i in taken:
                            continue
                        cost = self._match_cost(data, detections[i], frame_index)
                        if cost is None:
                            continue
                        # Favors confident, low-displacement links and recent tracks.
                        utility = 2.2 - cost + 0.05 * min(data['hits'], 10) / 10.0
                        search(k + 1, taken | {i}, score + utility, matches + [(tid, i)])
                search(0, set(), 0.0, [])

            matched_tracks = set()
            for tid, i in best_matches:
                ids[i] = tid
                matched_tracks.add(tid)
                self._record_observation(tid, detections[i], frame_index)

            unassigned = [i for i in indices if ids[i] is None]
            # When an old ID is temporarily missing, try reclaiming that slot
            # BEFORE allocating a fourth identity for the same physical robot.
            stale_tracks = sorted(
                [(tid, data) for tid, data in tracks
                 if tid not in matched_tracks and frame_index - data['last_frame'] >= 16],
                key=lambda pair: (pair[1]['last_frame'], pair[0])
            )
            for i in list(unassigned):
                if not stale_tracks:
                    break
                tid, _ = min(stale_tracks, key=lambda pair:
                             self._match_cost(pair[1], detections[i], frame_index, force=True))
                stale_tracks = [(t, data) for t, data in stale_tracks if t != tid]
                ids[i] = tid
                matched_tracks.add(tid)
                self._record_observation(tid, detections[i], frame_index)
                unassigned.remove(i)

            # Introduce a new slot only while fewer than three alliance IDs exist.
            while unassigned and len(tracks) < limit:
                i = unassigned.pop(0)
                tid = self.next_id
                self.next_id += 1
                det = detections[i]
                self.tracks[tid] = {
                    'label': label,
                    'cx': float(det['cx']), 'cy': float(det['cy']),
                    'vx': 0.0, 'vy': 0.0,
                    'w': float(det['x2'] - det['x1']),
                    'h': float(det['y2'] - det['y1']),
                    'box': dict(det), 'last_frame': frame_index, 'hits': 1
                }
                ids[i] = tid
                matched_tracks.add(tid)
                tracks.append((tid, self.tracks[tid]))

            # Important: NEVER discard unassigned/duplicate/9th+ YOLO boxes.
            # Match them to short-lived overflow tracks or create new overflow IDs.
            overflow_indices = [i for i in all_indices if ids[i] is None]
            if overflow_indices:
                extras = [(tid, data) for tid, data in self.tracks.items()
                          if data['label'] == label and data.get('overflow')]
                edges = []
                for tid, data in extras:
                    for i in overflow_indices:
                        cost = self._match_cost(data, detections[i], frame_index)
                        if cost is not None and cost <= 1.6:
                            edges.append((cost, tid, i))
                used_tids = set()
                used_indices = set()
                for cost, tid, i in sorted(edges):
                    if tid in used_tids or i in used_indices:
                        continue
                    ids[i] = tid
                    used_tids.add(tid)
                    used_indices.add(i)
                    self._record_observation(tid, detections[i], frame_index)

                for i in overflow_indices:
                    if ids[i] is not None:
                        continue
                    det = detections[i]
                    tid = self.next_id
                    self.next_id += 1
                    self.tracks[tid] = {
                        'label': label,
                        'cx': float(det['cx']), 'cy': float(det['cy']),
                        'vx': 0.0, 'vy': 0.0,
                        'w': float(det['x2'] - det['x1']),
                        'h': float(det['y2'] - det['y1']),
                        'box': dict(det), 'last_frame': frame_index,
                        'hits': 1, 'overflow': True,
                    }
                    ids[i] = tid

        # Do not silently filter an unknown class either. The main app only
        # passes red and blue candidates, so this catches integration mistakes.
        if any(tid is None for tid in ids):
            missing = [i for i, tid in enumerate(ids) if tid is None]
            raise RuntimeError(f'Tracker did not assign IDs to detections: {missing}')
        return ids
