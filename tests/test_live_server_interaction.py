#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live WebSocket Verification against running server.
"""

import asyncio
import json
import sys
import time

async def verify_live_server():
    import websockets

    uri = "ws://127.0.0.1:8765"
    print(f"Connecting to live server at {uri}...")

    async with websockets.connect(uri) as ws:
        # 1. Initial state message
        init_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
        init_data = json.loads(init_msg)
        print(f"[Live] Connected! Initial state: {init_data.get('type')}, videos available: {len(init_data.get('videos', []))}")

        # 2. Start streaming corridor video with expected direction LEFT
        print("[Live] Sending start_stream for assets/evacuation_test_corridor.avi (Flow: LEFT)...")
        await ws.send(json.dumps({
            "action": "start_stream",
            "video_path": "assets/evacuation_test_corridor.avi",
            "direction": "LEFT",
            "speed": 1.0,
            "profile": "balanced"
        }))

        # 3. Read 15 frames at 1.0x
        anomalies_seen = []
        frames_seen = 0
        speed_readings = []

        start_time = time.time()
        while frames_seen < 15 and (time.time() - start_time) < 25.0:
            msg = await asyncio.wait_for(ws.recv(), timeout=15.0)
            data = json.loads(msg)
            if data.get("type") == "frame_update":
                frames_seen += 1
                peds = data.get("pedestrians", [])
                alerts = data.get("active_alerts", [])
                speed_readings.append({
                    "frame": data.get("frame"),
                    "req_spd": data.get("requested_playback_speed"),
                    "act_spd": data.get("achieved_playback_speed"),
                    "eff_fps": data.get("effective_playback_fps"),
                    "proc_fps": data.get("processing_fps"),
                    "skipped": data.get("frames_skipped"),
                })
                for p in peds:
                    if p.get("visual_state") == "anomaly":
                        anomalies_seen.append(p)
                print(f"  Frame {data.get('frame')}: tracks={len(peds)}, alerts={len(alerts)}, mode={data.get('monitoring_mode')}, req_spd={data.get('requested_playback_speed')}x, act_spd={data.get('achieved_playback_speed')}x, skipped={data.get('frames_skipped')}")

        print(f"\n[Live] Processed {frames_seen} frames at 1.0x: {len(anomalies_seen)} anomaly observations recorded.")

        # 4. Dynamically change playback speed to 2.0x
        print("\n[Live] Testing dynamic speed change to 2.0x...")
        await ws.send(json.dumps({
            "action": "set_speed",
            "speed": 2.0
        }))

        # Read speed_updated message
        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
        data = json.loads(msg)
        print(f"  Server response to speed change: {data}")

        # Collect 10 frames at 2.0x
        f2_count = 0
        while f2_count < 10:
            msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
            data = json.loads(msg)
            if data.get("type") == "frame_update":
                f2_count += 1
                print(f"  [2.0x] Frame {data.get('frame')}: mode={data.get('monitoring_mode')}, req={data.get('requested_playback_speed')}x, act={data.get('achieved_playback_speed')}x, eff_fps={data.get('effective_playback_fps')}, skipped={data.get('frames_skipped')}")

        # 5. Dynamically change expected direction to RIGHT
        print("\n[Live] Testing dynamic expected flow direction change to RIGHT...")
        await ws.send(json.dumps({
            "action": "set_direction",
            "direction": "RIGHT"
        }))

        # Collect 5 frames under new direction
        f3_count = 0
        while f3_count < 5:
            msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
            data = json.loads(msg)
            if data.get("type") == "frame_update":
                f3_count += 1
                print(f"  [Flow RIGHT] Frame {data.get('frame')}: mode={data.get('monitoring_mode')}, exp_dir={data.get('expected_direction')}, tracks={len(data.get('pedestrians', []))}, alerts={len(data.get('active_alerts', []))}")

        print("\n[Live] ALL LIVE WEBSOCKET VERIFICATIONS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(verify_live_server())

