#!/usr/bin/env python3
"""
Feature extraction script for bot detection.
Extracts engineering features from interaction JSON files.
"""

import json
import os
import csv
import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq
from collections import defaultdict
from pathlib import Path
import math


def compute_statistics(values):
    """Compute mean, median, std, skew, kurtosis for a list of values."""
    if len(values) == 0:
        return 0, 0, 0, 0, 0
    values = np.array(values)
    return (
        np.mean(values),
        np.median(values),
        np.std(values),
        stats.skew(values),
        stats.kurtosis(values)
    )


def compute_percentiles(values, percentiles=[10, 25, 50, 75, 90, 95, 99]):
    """Compute percentiles for a list of values."""
    if len(values) == 0:
        return {p: 0 for p in percentiles}
    return {p: np.percentile(values, p) for p in percentiles}


def compute_entropy(values, bins=20):
    """Compute entropy of a distribution."""
    if len(values) == 0:
        return 0
    hist, _ = np.histogram(values, bins=bins)
    hist = hist[hist > 0]  # Remove zeros
    if len(hist) == 0:
        return 0
    probs = hist / hist.sum()
    return -np.sum(probs * np.log2(probs + 1e-10))


def extract_time_features(events):
    """Extract time-based features."""
    if len(events) < 2:
        return {}
    
    timestamps = [e['timestamp'] for e in events]
    dt = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
    
    if len(dt) == 0:
        return {}
    
    dt = np.array(dt) / 1000.0  # Convert to seconds
    
    mean_dt, median_dt, std_dt, skew_dt, kurt_dt = compute_statistics(dt)
    
    # Burstiness index
    if mean_dt > 0:
        burstiness = (std_dt - mean_dt) / (std_dt + mean_dt + 1e-10)
    else:
        burstiness = 0
    
    # Longest pause
    longest_pause = np.max(dt) if len(dt) > 0 else 0
    
    # Percentiles
    dt_percentiles = compute_percentiles(dt)
    
    return {
        'time_inter_event_mean': mean_dt,
        'time_inter_event_median': median_dt,
        'time_inter_event_std': std_dt,
        'time_inter_event_skew': skew_dt,
        'time_inter_event_kurtosis': kurt_dt,
        'time_burstiness': burstiness,
        'time_longest_pause': longest_pause,
        'time_inter_event_p10': dt_percentiles[10],
        'time_inter_event_p25': dt_percentiles[25],
        'time_inter_event_p75': dt_percentiles[75],
        'time_inter_event_p90': dt_percentiles[90],
        'time_inter_event_p95': dt_percentiles[95],
        'time_inter_event_p99': dt_percentiles[99],
    }


def extract_mouse_features(events):
    """Extract mouse movement features."""
    mouse_events = [e for e in events if e.get('action') == 'mouse_move' and 'x' in e and 'y' in e]
    
    if len(mouse_events) < 2:
        return {}
    
    features = {}
    
    # Extract positions and timestamps
    positions = [(e['x'], e['y']) for e in mouse_events]
    timestamps = [e['timestamp'] for e in mouse_events]
    
    # Compute distances and time deltas
    distances = []
    dt = []
    speeds = []
    
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        dist = math.sqrt(dx*dx + dy*dy)
        dt_i = (timestamps[i] - timestamps[i-1]) / 1000.0  # seconds
        
        distances.append(dist)
        dt.append(dt_i)
        
        if dt_i > 0:
            speeds.append(dist / dt_i)
        else:
            speeds.append(0)
    
    if len(speeds) == 0:
        return {}
    
    # 6. Speed statistics
    speed_mean, speed_median, speed_std, speed_skew, speed_kurt = compute_statistics(speeds)
    speed_percentiles = compute_percentiles(speeds)
    
    features.update({
        'mouse_speed_mean': speed_mean,
        'mouse_speed_median': speed_median,
        'mouse_speed_std': speed_std,
        'mouse_speed_skew': speed_skew,
        'mouse_speed_kurtosis': speed_kurt,
        'mouse_speed_p10': speed_percentiles[10],
        'mouse_speed_p25': speed_percentiles[25],
        'mouse_speed_p75': speed_percentiles[75],
        'mouse_speed_p90': speed_percentiles[90],
        'mouse_speed_p95': speed_percentiles[95],
        'mouse_speed_p99': speed_percentiles[99],
    })
    
    # 7. Acceleration & jerk
    if len(speeds) >= 2:
        accelerations = []
        for i in range(1, len(speeds)):
            if dt[i] > 0:
                acc = (speeds[i] - speeds[i-1]) / dt[i]
                accelerations.append(acc)
        
        if len(accelerations) > 0:
            acc_mean, acc_median, acc_std, acc_skew, acc_kurt = compute_statistics(accelerations)
            features.update({
                'mouse_acceleration_mean': acc_mean,
                'mouse_acceleration_std': acc_std,
                'mouse_acceleration_skew': acc_skew,
            })
            
            # Jerk (3rd derivative)
            if len(accelerations) >= 2:
                jerks = []
                for i in range(1, len(accelerations)):
                    if dt[i] > 0:
                        jerk = (accelerations[i] - accelerations[i-1]) / dt[i]
                        jerks.append(jerk)
                
                if len(jerks) > 0:
                    jerk_mean, jerk_median, jerk_std, jerk_skew, jerk_kurt = compute_statistics(jerks)
                    features.update({
                        'mouse_jerk_mean': jerk_mean,
                        'mouse_jerk_std': jerk_std,
                    })
    
    # 8. Curvature
    if len(positions) >= 3:
        curvatures = []
        for i in range(1, len(positions) - 1):
            p0 = positions[i-1]
            p1 = positions[i]
            p2 = positions[i+1]
            
            dt1 = (timestamps[i] - timestamps[i-1]) / 1000.0
            dt2 = (timestamps[i+1] - timestamps[i]) / 1000.0
            
            if dt1 > 0 and dt2 > 0:
                # First derivatives
                x1_prime = (p1[0] - p0[0]) / dt1
                y1_prime = (p1[1] - p0[1]) / dt1
                x2_prime = (p2[0] - p1[0]) / dt2
                y2_prime = (p2[1] - p1[1]) / dt2
                
                # Second derivatives (approximate)
                dt_avg = (dt1 + dt2) / 2.0
                if dt_avg > 0:
                    x_double_prime = (x2_prime - x1_prime) / dt_avg
                    y_double_prime = (y2_prime - y1_prime) / dt_avg
                    
                    # Curvature formula
                    speed_sq = x1_prime**2 + y1_prime**2
                    if speed_sq > 0:
                        curvature = abs(x1_prime * y_double_prime - y1_prime * x_double_prime) / (speed_sq ** 1.5)
                        curvatures.append(curvature)
        
        if len(curvatures) > 0:
            curv_mean, curv_median, curv_std, curv_skew, curv_kurt = compute_statistics(curvatures)
            features.update({
                'mouse_curvature_mean': curv_mean,
                'mouse_curvature_std': curv_std,
            })
            
            # 9. Curvature-velocity correlation
            if len(curvatures) > 1 and len(speeds) > 1:
                min_len = min(len(curvatures), len(speeds))
                if min_len > 1:
                    corr = np.corrcoef(curvatures[:min_len], speeds[:min_len])[0, 1]
                    features['mouse_curvature_velocity_corr'] = corr if not np.isnan(corr) else 0
    
    # 10. Path straightness
    if len(positions) >= 2:
        total_path_length = sum(distances)
        start_end_dist = math.sqrt(
            (positions[-1][0] - positions[0][0])**2 + 
            (positions[-1][1] - positions[0][1])**2
        )
        if start_end_dist > 0:
            features['mouse_path_straightness'] = total_path_length / start_end_dist
        else:
            features['mouse_path_straightness'] = 1.0
    
    # 11. Direction changes
    if len(positions) >= 3:
        angles = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            angle = math.atan2(dy, dx)
            angles.append(angle)
        
        direction_changes = 0
        for i in range(1, len(angles)):
            angle_diff = angles[i] - angles[i-1]
            # Normalize to [-pi, pi]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            # Count significant direction changes (more than 45 degrees)
            if abs(angle_diff) > math.pi / 4:
                direction_changes += 1
        
        features['mouse_direction_changes'] = direction_changes
    
    # 12. Movement entropy
    if len(positions) >= 2:
        angles = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            angle = math.atan2(dy, dx)
            angles.append(angle)
        
        if len(angles) > 0:
            features['mouse_movement_entropy'] = compute_entropy(angles)
    
    # 13. Micro-jitter frequency spectrum
    if len(positions) >= 4:
        # Compute deltas
        x_deltas = [positions[i][0] - positions[i-1][0] for i in range(1, len(positions))]
        y_deltas = [positions[i][1] - positions[i-1][1] for i in range(1, len(positions))]
        
        # FFT on combined signal
        combined = np.array([math.sqrt(x**2 + y**2) for x, y in zip(x_deltas, y_deltas)])
        
        if len(combined) > 1:
            # Sample rate (approximate)
            avg_dt = np.mean(dt) if len(dt) > 0 else 0.016  # Default 60Hz
            if avg_dt > 0:
                sample_rate = 1.0 / avg_dt
                fft_vals = fft(combined)
                freqs = fftfreq(len(combined), 1.0 / sample_rate)
                
                # Power spectrum
                power = np.abs(fft_vals) ** 2
                
                # Find peak in 8-12 Hz range (human tremor)
                mask = (freqs >= 8) & (freqs <= 12)
                if np.any(mask):
                    peak_power = np.max(power[mask])
                    features['mouse_tremor_peak_power'] = peak_power
                else:
                    features['mouse_tremor_peak_power'] = 0
    
    return features


def extract_click_features(events):
    """Extract click-related features."""
    click_events = [e for e in events if e.get('action') == 'click']
    mouse_events = [e for e in events if e.get('action') == 'mouse_move' and 'x' in e and 'y' in e]
    
    if len(click_events) == 0:
        return {}
    
    features = {}
    
    # 15. Click hesitation time
    hesitation_times = []
    for click in click_events:
        click_time = click['timestamp']
        click_x = click.get('x', 0)
        click_y = click.get('y', 0)
        
        # Find last mouse move before click
        last_move_time = None
        for mouse in reversed(mouse_events):
            if mouse['timestamp'] < click_time:
                last_move_time = mouse['timestamp']
                break
        
        if last_move_time is not None:
            hesitation = (click_time - last_move_time) / 1000.0  # seconds
            hesitation_times.append(hesitation)
    
    if len(hesitation_times) > 0:
        hes_mean, hes_median, hes_std, hes_skew, hes_kurt = compute_statistics(hesitation_times)
        features.update({
            'click_hesitation_mean': hes_mean,
            'click_hesitation_median': hes_median,
            'click_hesitation_std': hes_std,
        })
    
    # 16. Pre-click path length
    pre_click_paths = []
    for click in click_events:
        click_time = click['timestamp']
        click_x = click.get('x', 0)
        click_y = click.get('y', 0)
        
        # Find mouse moves in last 500ms before click
        path_length = 0
        prev_pos = None
        for mouse in reversed(mouse_events):
            if mouse['timestamp'] < click_time and (click_time - mouse['timestamp']) <= 500:
                if prev_pos is not None:
                    dx = mouse['x'] - prev_pos[0]
                    dy = mouse['y'] - prev_pos[1]
                    path_length += math.sqrt(dx*dx + dy*dy)
                prev_pos = (mouse['x'], mouse['y'])
            elif mouse['timestamp'] < click_time - 500:
                break
        
        pre_click_paths.append(path_length)
    
    if len(pre_click_paths) > 0:
        path_mean, path_median, path_std, _, _ = compute_statistics(pre_click_paths)
        features.update({
            'click_pre_path_mean': path_mean,
            'click_pre_path_median': path_median,
        })
    
    # 14. Target approach velocity profile
    approach_speed_ratios = []
    for click in click_events:
        click_time = click['timestamp']
        click_x = click.get('x', 0)
        click_y = click.get('y', 0)
        
        # Get mouse moves in last 200ms before click
        recent_moves = []
        for mouse in reversed(mouse_events):
            if mouse['timestamp'] < click_time and (click_time - mouse['timestamp']) <= 200:
                recent_moves.insert(0, mouse)
            elif mouse['timestamp'] < click_time - 200:
                break
        
        if len(recent_moves) >= 2:
            # Compute velocities approaching target
            approach_velocities = []
            for i in range(1, len(recent_moves)):
                dx = recent_moves[i]['x'] - recent_moves[i-1]['x']
                dy = recent_moves[i]['y'] - recent_moves[i-1]['y']
                dt = (recent_moves[i]['timestamp'] - recent_moves[i-1]['timestamp']) / 1000.0
                
                if dt > 0:
                    dist_to_target = math.sqrt(
                        (recent_moves[i]['x'] - click_x)**2 + 
                        (recent_moves[i]['y'] - click_y)**2
                    )
                    speed = math.sqrt(dx*dx + dy*dy) / dt
                    approach_velocities.append((dist_to_target, speed))
            
            # Check if velocity decreases as approaching (human behavior)
            if len(approach_velocities) >= 2:
                # Sort by distance (farthest to closest)
                approach_velocities.sort(key=lambda x: x[0])
                early_speed = approach_velocities[0][1]  # Farthest
                late_speed = approach_velocities[-1][1]  # Closest
                if early_speed > 0:
                    speed_ratio = late_speed / early_speed
                    approach_speed_ratios.append(speed_ratio)
    
    if len(approach_speed_ratios) > 0:
        features['click_approach_speed_ratio_mean'] = np.mean(approach_speed_ratios)
        features['click_approach_speed_ratio_median'] = np.median(approach_speed_ratios)
    
    # 17. Click entropy
    if len(click_events) > 0:
        click_positions = [(e.get('x', 0), e.get('y', 0)) for e in click_events]
        x_coords = [p[0] for p in click_positions]
        y_coords = [p[1] for p in click_positions]
        
        if len(x_coords) > 0:
            features['click_x_entropy'] = compute_entropy(x_coords)
            features['click_y_entropy'] = compute_entropy(y_coords)
    
    # 18. Double-click detection
    if len(click_events) >= 2:
        double_click_times = []
        for i in range(1, len(click_events)):
            dt = (click_events[i]['timestamp'] - click_events[i-1]['timestamp']) / 1000.0
            # Typical double-click is 200-500ms
            if 0.2 <= dt <= 0.5:
                double_click_times.append(dt)
        
        if len(double_click_times) > 0:
            features['click_double_click_count'] = len(double_click_times)
            features['click_double_click_mean_time'] = np.mean(double_click_times)
    
    return features


def extract_scroll_features(events):
    """Extract scrolling features."""
    # Note: The dataset may not have scroll events, but we'll handle it
    scroll_events = [e for e in events if e.get('action') == 'scroll']
    
    if len(scroll_events) == 0:
        return {}
    
    features = {}
    
    timestamps = [e['timestamp'] for e in scroll_events]
    deltas = [e.get('delta', 0) for e in scroll_events]
    
    if len(scroll_events) < 2:
        return {}
    
    # 19. Scroll velocity & acceleration
    scroll_velocities = []
    dt = []
    for i in range(1, len(scroll_events)):
        dt_i = (timestamps[i] - timestamps[i-1]) / 1000.0
        if dt_i > 0:
            velocity = abs(deltas[i]) / dt_i
            scroll_velocities.append(velocity)
            dt.append(dt_i)
    
    if len(scroll_velocities) > 0:
        vel_mean, vel_median, vel_std, vel_skew, vel_kurt = compute_statistics(scroll_velocities)
        features.update({
            'scroll_velocity_mean': vel_mean,
            'scroll_velocity_std': vel_std,
        })
        
        # Acceleration
        if len(scroll_velocities) >= 2:
            accelerations = []
            for i in range(1, len(scroll_velocities)):
                if dt[i] > 0:
                    acc = (scroll_velocities[i] - scroll_velocities[i-1]) / dt[i]
                    accelerations.append(acc)
            
            if len(accelerations) > 0:
                acc_mean, _, acc_std, _, _ = compute_statistics(accelerations)
                features.update({
                    'scroll_acceleration_mean': acc_mean,
                    'scroll_acceleration_std': acc_std,
                })
    
    # 20. Scroll burst structure
    if len(dt) > 0:
        scroll_burstiness = compute_statistics(dt)[0]  # Use mean inter-scroll time
        features['scroll_burstiness'] = scroll_burstiness
    
    # 21. Scroll direction entropy
    if len(deltas) > 0:
        directions = [1 if d > 0 else -1 for d in deltas]
        features['scroll_direction_entropy'] = compute_entropy(directions)
    
    # 22. Time spent stable (no scrolling)
    if len(events) > 0:
        total_time = (events[-1]['timestamp'] - events[0]['timestamp']) / 1000.0
        scroll_time = sum(dt) if len(dt) > 0 else 0
        if total_time > 0:
            features['scroll_idle_ratio'] = 1.0 - (scroll_time / total_time)
    
    return features


def extract_keystroke_features(events):
    """Extract keystroke features."""
    key_events = [e for e in events if e.get('action') == 'keypress']
    
    if len(key_events) == 0:
        return {}
    
    features = {}
    
    timestamps = [e['timestamp'] for e in key_events]
    
    # Note: Feature 23 (Key hold duration) requires keydown/keyup events,
    # but we only have keypress events, so it cannot be computed.
    
    # 24. Inter-key latencies
    if len(timestamps) >= 2:
        inter_key_times = [(timestamps[i] - timestamps[i-1]) / 1000.0 for i in range(1, len(timestamps))]
        
        if len(inter_key_times) > 0:
            lat_mean, lat_median, lat_std, lat_skew, lat_kurt = compute_statistics(inter_key_times)
            features.update({
                'keystroke_inter_key_mean': lat_mean,
                'keystroke_inter_key_median': lat_median,
                'keystroke_inter_key_std': lat_std,
                'keystroke_inter_key_skew': lat_skew,
            })
    
    # 25. Error + correction patterns
    # Note: We don't have key information, so we can't detect backspaces directly
    # But we can look for patterns in timing that suggest corrections
    
    # 26. Typing burstiness
    if len(timestamps) >= 2:
        inter_key_times = [(timestamps[i] - timestamps[i-1]) / 1000.0 for i in range(1, len(timestamps))]
        if len(inter_key_times) > 0:
            mean_lat = np.mean(inter_key_times)
            std_lat = np.std(inter_key_times)
            if mean_lat > 0:
                typing_burstiness = (std_lat - mean_lat) / (std_lat + mean_lat + 1e-10)
                features['keystroke_burstiness'] = typing_burstiness
    
    return features


def extract_session_features(events):
    """Extract session-level aggregate features."""
    features = {}
    
    # 31. Total event counts
    event_types = defaultdict(int)
    for e in events:
        event_types[e.get('action', 'unknown')] += 1
    
    features.update({
        'session_total_events': len(events),
        'session_mouse_move_count': event_types['mouse_move'],
        'session_click_count': event_types['click'],
        'session_keypress_count': event_types['keypress'],
        'session_scroll_count': event_types['scroll'],
    })
    
    # 32. Event type frequency distribution
    if len(events) > 0:
        for event_type, count in event_types.items():
            features[f'session_{event_type}_ratio'] = count / len(events)
    
    # 33. Idle time ratio
    # Idle time = time spent in gaps longer than 1 second (reading/thinking time)
    if len(events) >= 2:
        total_time = (events[-1]['timestamp'] - events[0]['timestamp']) / 1000.0
        idle_threshold = 1.0  # 1 second
        idle_time = 0
        
        for i in range(1, len(events)):
            gap = (events[i]['timestamp'] - events[i-1]['timestamp']) / 1000.0
            if gap > idle_threshold:
                idle_time += gap - idle_threshold  # Only count time beyond threshold
        
        if total_time > 0:
            features['session_idle_ratio'] = idle_time / total_time
        else:
            features['session_idle_ratio'] = 0
    
    # 34. Unique element interactions
    unique_elements = set()
    for e in events:
        if 'target' in e and isinstance(e['target'], dict):
            element_id = e['target'].get('tag', '') + str(e['target'].get('id', ''))
            unique_elements.add(element_id)
    
    features['session_unique_elements'] = len(unique_elements)
    
    # 35. Entropy across all feature distributions
    # We'll compute entropy for various distributions
    if len(events) >= 2:
        timestamps = [e['timestamp'] for e in events]
        dt = [(timestamps[i] - timestamps[i-1]) / 1000.0 for i in range(1, len(timestamps))]
        if len(dt) > 0:
            features['session_time_entropy'] = compute_entropy(dt)
    
    return features


def extract_features_from_session(session_events):
    """Extract all features from a session."""
    features = {}
    
    # Sort events by timestamp
    session_events = sorted(session_events, key=lambda x: x.get('timestamp', 0))
    
    # Extract features from each category
    features.update(extract_time_features(session_events))
    features.update(extract_mouse_features(session_events))
    features.update(extract_click_features(session_events))
    features.update(extract_scroll_features(session_events))
    features.update(extract_keystroke_features(session_events))
    features.update(extract_session_features(session_events))
    
    return features


def process_json_file(file_path, label=None):
    """Process a single JSON file and extract features for all sessions."""
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    all_features = []
    
    for user_id, sessions in data.items():
        for session_id, events in sessions.items():
            if not isinstance(events, list) or len(events) == 0:
                continue
            
            features = extract_features_from_session(events)
            features['user_id'] = user_id
            features['session_id'] = session_id
            features['source_file'] = os.path.basename(file_path)
            
            # Add label if provided
            if label is not None:
                features['label'] = label
            
            all_features.append(features)
    
    return all_features


def main():
    """Main function to process all JSON files and create CSV."""
    # Use real human data from ../DATA and synthetic bot data from ../Synthesizer/output
    base_dir = Path(__file__).parent.parent  # Go up from CLASSIFIER to Thesis
    human_data_dir = base_dir / 'DATA'
    bot_data_dir = base_dir / 'Synthesizer' / 'output' / 'v1'
    output_file = Path(__file__).parent / 'features.csv'
    
    json_files = []
    labels = []
    
    # Load human data (label = 1)
    if human_data_dir.exists():
        print(f"Loading human data from {human_data_dir}...")
        human_count = 0
        for json_file in human_data_dir.glob('*.json'):
            json_files.append(json_file)
            labels.append(1)
            human_count += 1
        print(f"Found {human_count} human data files")
    else:
        print(f"Warning: Human data directory not found: {human_data_dir}")
    
    # Load bot data (label = 0)
    if bot_data_dir.exists():
        print(f"Loading bot data from {bot_data_dir}...")
        bot_count = 0
        for json_file in bot_data_dir.glob('*.json'):
            json_files.append(json_file)
            labels.append(0)
            bot_count += 1
        print(f"Found {bot_count} bot data files")
    else:
        print(f"Warning: Bot data directory not found: {bot_data_dir}")
    
    if len(json_files) == 0:
        print("No JSON files found in data directory!")
        return
    
    print(f"Found {len(json_files)} JSON files to process.")
    
    # Process all files
    all_features = []
    for json_file, label in zip(json_files, labels):
        try:
            features_list = process_json_file(json_file, label=label)
            all_features.extend(features_list)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    if len(all_features) == 0:
        print("No features extracted!")
        return
    
    # Get all unique feature names
    all_feature_names = set()
    for feat_dict in all_features:
        all_feature_names.update(feat_dict.keys())
    
    # Sort feature names (put metadata first)
    metadata_fields = ['user_id', 'session_id', 'source_file']
    if 'label' in all_feature_names:
        metadata_fields.append('label')
    feature_fields = sorted([f for f in all_feature_names if f not in metadata_fields])
    fieldnames = metadata_fields + feature_fields
    
    # Write to CSV
    print(f"Writing {len(all_features)} sessions to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for feat_dict in all_features:
            # Fill missing features with 0
            row = {field: feat_dict.get(field, 0) for field in fieldnames}
            writer.writerow(row)
    
    print(f"Done! Extracted {len(feature_fields)} features from {len(all_features)} sessions.")
    print(f"Output saved to: {output_file}")
    
    # Print label distribution
    if 'label' in all_feature_names:
        label_counts = defaultdict(int)
        for feat_dict in all_features:
            label = feat_dict.get('label')
            if label is not None:
                label_counts[label] += 1
        print(f"\nLabel distribution:")
        for label, count in sorted(label_counts.items()):
            label_name = "Human" if label == 1 else "Bot" if label == 0 else "Unknown"
            print(f"  {label_name} (label={label}): {count} sessions")


if __name__ == '__main__':
    main()

