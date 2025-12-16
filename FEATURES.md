# Bot Detection Features Documentation

This document provides a comprehensive explanation of all features extracted from user interaction data for bot detection. The features are organized into six main categories: Time-Based, Mouse Movement, Click, Scroll, Keystroke, and Session-Level features.

---

## 1. Time-Based Features

These features analyze the temporal patterns of events, capturing the rhythm and timing characteristics of user interactions.

### `time_inter_event_mean`
- **Description**: Mean time between consecutive events
- **Units**: Seconds
- **Purpose**: Captures the average pace of interactions. Humans typically have more variable pacing compared to bots.

### `time_inter_event_median`
- **Description**: Median time between consecutive events
- **Units**: Seconds
- **Purpose**: Provides a robust measure of central tendency, less affected by outliers than the mean.

### `time_inter_event_std`
- **Description**: Standard deviation of inter-event times
- **Units**: Seconds
- **Purpose**: Measures variability in timing. Higher values indicate more irregular, human-like behavior.

### `time_inter_event_skew`
- **Description**: Skewness of inter-event time distribution
- **Purpose**: Measures asymmetry in timing patterns. Positive skew indicates occasional long pauses (human behavior).

### `time_inter_event_kurtosis`
- **Description**: Kurtosis of inter-event time distribution
- **Purpose**: Measures the "tailedness" of the distribution. High kurtosis indicates more extreme values (long pauses or rapid bursts).

### `time_burstiness`
- **Description**: Burstiness index calculated as `(std - mean) / (std + mean)`
- **Range**: -1 to 1
- **Purpose**: Quantifies how bursty the interaction pattern is. Values closer to 1 indicate more bursty behavior (typical of humans), while values closer to -1 indicate more regular patterns (typical of bots).

### `time_longest_pause`
- **Description**: Maximum time gap between any two consecutive events
- **Units**: Seconds
- **Purpose**: Captures the longest thinking/reading pause, which is characteristic of human behavior.

### `time_inter_event_p10`, `time_inter_event_p25`, `time_inter_event_p75`, `time_inter_event_p90`, `time_inter_event_p95`, `time_inter_event_p99`
- **Description**: Percentiles (10th, 25th, 75th, 90th, 95th, 99th) of inter-event times
- **Units**: Seconds
- **Purpose**: Provides detailed distribution information, capturing both typical and extreme timing behaviors.

---

## 2. Mouse Movement Features

These features analyze mouse movement patterns, capturing the natural dynamics and micro-movements characteristic of human motor control.

### Speed Features

#### `mouse_speed_mean`
- **Description**: Average mouse movement speed
- **Units**: Pixels per second
- **Purpose**: Humans typically have more variable speeds compared to bots.

#### `mouse_speed_median`
- **Description**: Median mouse movement speed
- **Units**: Pixels per second
- **Purpose**: Robust measure of typical movement speed.

#### `mouse_speed_std`
- **Description**: Standard deviation of mouse speeds
- **Units**: Pixels per second
- **Purpose**: Measures speed variability. Higher values indicate more natural, human-like movement.

#### `mouse_speed_skew`
- **Description**: Skewness of speed distribution
- **Purpose**: Captures asymmetry in speed patterns.

#### `mouse_speed_kurtosis`
- **Description**: Kurtosis of speed distribution
- **Purpose**: Measures the presence of extreme speed values.

#### `mouse_speed_p10`, `mouse_speed_p25`, `mouse_speed_p75`, `mouse_speed_p90`, `mouse_speed_p95`, `mouse_speed_p99`
- **Description**: Percentiles of mouse speed distribution
- **Units**: Pixels per second
- **Purpose**: Detailed speed distribution information.

### Acceleration and Jerk Features

#### `mouse_acceleration_mean`
- **Description**: Average rate of change of mouse speed
- **Units**: Pixels per second²
- **Purpose**: Humans show more variable acceleration patterns. Bots often have more uniform acceleration.

#### `mouse_acceleration_std`
- **Description**: Standard deviation of acceleration
- **Units**: Pixels per second²
- **Purpose**: Measures acceleration variability.

#### `mouse_acceleration_skew`
- **Description**: Skewness of acceleration distribution
- **Purpose**: Captures asymmetry in acceleration patterns.

#### `mouse_jerk_mean`
- **Description**: Average rate of change of acceleration (3rd derivative of position)
- **Units**: Pixels per second³
- **Purpose**: Jerk is a key indicator of smoothness. Human movements have natural jerk, while bot movements are often unnaturally smooth.

#### `mouse_jerk_std`
- **Description**: Standard deviation of jerk
- **Units**: Pixels per second³
- **Purpose**: Measures jerk variability.

### Curvature Features

#### `mouse_curvature_mean`
- **Description**: Average curvature of mouse path
- **Purpose**: Humans rarely move in perfectly straight lines. Curvature captures the natural arc of human movements.

#### `mouse_curvature_std`
- **Description**: Standard deviation of curvature
- **Purpose**: Measures variability in path curvature.

#### `mouse_curvature_velocity_corr`
- **Description**: Correlation between curvature and velocity
- **Range**: -1 to 1
- **Purpose**: In human movements, there's often a relationship between speed and curvature (slower around curves). Bots may lack this natural correlation.

### Path Characteristics

#### `mouse_path_straightness`
- **Description**: Ratio of total path length to straight-line distance from start to end
- **Range**: ≥ 1.0 (1.0 = perfectly straight)
- **Purpose**: Humans rarely take perfectly straight paths. Values significantly above 1.0 indicate more natural, meandering paths.

#### `mouse_direction_changes`
- **Description**: Count of significant direction changes (>45 degrees)
- **Purpose**: Humans make frequent small corrections. Bots may have fewer or more abrupt direction changes.

#### `mouse_movement_entropy`
- **Description**: Entropy of movement direction angles
- **Purpose**: Measures the randomness/diversity of movement directions. Higher entropy indicates more varied, human-like movement patterns.

### Micro-Movements

#### `mouse_tremor_peak_power`
- **Description**: Peak power in the 8-12 Hz frequency range (human tremor range)
- **Purpose**: Humans exhibit natural hand tremor in the 8-12 Hz range. This feature detects the presence of this micro-movement, which is difficult for bots to replicate.

---

## 3. Click Features

These features analyze clicking behavior, capturing the hesitation, approach patterns, and click characteristics that distinguish human from bot behavior.

### Click Hesitation

#### `click_hesitation_mean`
- **Description**: Average time between last mouse movement and click
- **Units**: Seconds
- **Purpose**: Humans typically pause briefly before clicking (hesitation). Bots often click immediately after reaching the target.

#### `click_hesitation_median`
- **Description**: Median hesitation time
- **Units**: Seconds
- **Purpose**: Robust measure of typical hesitation.

#### `click_hesitation_std`
- **Description**: Standard deviation of hesitation times
- **Units**: Seconds
- **Purpose**: Measures variability in hesitation patterns.

### Pre-Click Path

#### `click_pre_path_mean`
- **Description**: Average path length traveled in the 500ms before a click
- **Units**: Pixels
- **Purpose**: Humans often make small adjustments before clicking. Bots may have shorter or more direct paths.

#### `click_pre_path_median`
- **Description**: Median pre-click path length
- **Units**: Pixels
- **Purpose**: Robust measure of typical pre-click movement.

### Target Approach

#### `click_approach_speed_ratio_mean`
- **Description**: Average ratio of speed when closest to target vs. speed when farthest from target (in last 200ms)
- **Purpose**: Humans typically slow down as they approach a target (Fitts' law). This ratio captures this deceleration pattern. Values < 1.0 indicate natural deceleration.

#### `click_approach_speed_ratio_median`
- **Description**: Median approach speed ratio
- **Purpose**: Robust measure of typical approach behavior.

### Click Distribution

#### `click_x_entropy`
- **Description**: Entropy of click X-coordinates
- **Purpose**: Measures the randomness of click positions. Humans click in more varied locations.

#### `click_y_entropy`
- **Description**: Entropy of click Y-coordinates
- **Purpose**: Measures the randomness of click positions vertically.

### Double-Click Detection

#### `click_double_click_count`
- **Description**: Number of double-clicks detected (clicks 200-500ms apart)
- **Purpose**: Humans sometimes double-click by accident or habit. Bots rarely exhibit this behavior.

#### `click_double_click_mean_time`
- **Description**: Average time between clicks in double-click pairs
- **Units**: Seconds
- **Purpose**: Typical double-click timing is around 200-500ms.

---

## 4. Scroll Features

These features analyze scrolling behavior, capturing the natural rhythm and patterns of human scrolling.

### Scroll Velocity

#### `scroll_velocity_mean`
- **Description**: Average scrolling velocity
- **Units**: Pixels per second
- **Purpose**: Humans scroll at variable speeds. Bots may scroll at more constant rates.

#### `scroll_velocity_std`
- **Description**: Standard deviation of scroll velocity
- **Units**: Pixels per second
- **Purpose**: Measures variability in scrolling speed.

### Scroll Acceleration

#### `scroll_acceleration_mean`
- **Description**: Average rate of change of scroll velocity
- **Units**: Pixels per second²
- **Purpose**: Captures how smoothly scrolling speed changes.

#### `scroll_acceleration_std`
- **Description**: Standard deviation of scroll acceleration
- **Units**: Pixels per second²
- **Purpose**: Measures acceleration variability.

### Scroll Patterns

#### `scroll_burstiness`
- **Description**: Mean time between scroll events
- **Units**: Seconds
- **Purpose**: Captures the bursty nature of scrolling. Humans scroll in bursts with pauses.

#### `scroll_direction_entropy`
- **Description**: Entropy of scroll directions (up/down)
- **Purpose**: Measures the randomness of scroll direction changes. Humans scroll both up and down naturally.

#### `scroll_idle_ratio`
- **Description**: Ratio of time spent not scrolling (gaps > 1 second) to total session time
- **Range**: 0 to 1
- **Purpose**: Humans pause to read content. Higher values indicate more natural reading behavior.

---

## 5. Keystroke Features

These features analyze typing patterns, capturing the rhythm and timing characteristics of human typing.

### Inter-Key Timing

#### `keystroke_inter_key_mean`
- **Description**: Average time between consecutive keystrokes
- **Units**: Seconds
- **Purpose**: Humans type at variable speeds. Bots may type at unnaturally constant rates.

#### `keystroke_inter_key_median`
- **Description**: Median inter-key time
- **Units**: Seconds
- **Purpose**: Robust measure of typical typing speed.

#### `keystroke_inter_key_std`
- **Description**: Standard deviation of inter-key times
- **Units**: Seconds
- **Purpose**: Measures typing rhythm variability. Higher values indicate more natural, human-like typing.

#### `keystroke_inter_key_skew`
- **Description**: Skewness of inter-key time distribution
- **Purpose**: Captures asymmetry in typing patterns. Positive skew indicates occasional long pauses (thinking/correction).

### Typing Patterns

#### `keystroke_burstiness`
- **Description**: Burstiness index for typing, calculated as `(std - mean) / (std + mean)`
- **Range**: -1 to 1
- **Purpose**: Quantifies how bursty the typing pattern is. Humans type in bursts with pauses, while bots may type more continuously.

---

## 6. Session-Level Features

These features provide aggregate statistics about the entire session, capturing overall interaction patterns.

### Event Counts

#### `session_total_events`
- **Description**: Total number of events in the session
- **Purpose**: Overall activity level indicator.

#### `session_mouse_move_count`
- **Description**: Total number of mouse movement events
- **Purpose**: Measures mouse activity level.

#### `session_click_count`
- **Description**: Total number of click events
- **Purpose**: Measures clicking activity.

#### `session_keypress_count`
- **Description**: Total number of keypress events
- **Purpose**: Measures typing activity.

#### `session_scroll_count`
- **Description**: Total number of scroll events
- **Purpose**: Measures scrolling activity.

### Event Ratios

#### `session_mouse_move_ratio`
- **Description**: Proportion of mouse movement events to total events
- **Range**: 0 to 1
- **Purpose**: Relative frequency of mouse movements.

#### `session_click_ratio`
- **Description**: Proportion of click events to total events
- **Range**: 0 to 1
- **Purpose**: Relative frequency of clicks.

#### `session_keypress_ratio`
- **Description**: Proportion of keypress events to total events
- **Range**: 0 to 1
- **Purpose**: Relative frequency of typing.

#### `session_scroll_ratio`
- **Description**: Proportion of scroll events to total events
- **Range**: 0 to 1
- **Purpose**: Relative frequency of scrolling.

#### `session_<event_type>_ratio`
- **Description**: Proportion of any other event type to total events
- **Range**: 0 to 1
- **Purpose**: Captures the relative frequency of any event type in the session.

### Session Characteristics

#### `session_idle_ratio`
- **Description**: Proportion of time spent in idle periods (gaps > 1 second) to total session time
- **Range**: 0 to 1
- **Purpose**: Humans pause to read, think, or process information. Higher values indicate more natural human behavior.

#### `session_unique_elements`
- **Description**: Number of unique DOM elements interacted with
- **Purpose**: Measures the diversity of interactions. Humans explore more, while bots may focus on specific elements.

#### `session_time_entropy`
- **Description**: Entropy of inter-event time distribution
- **Purpose**: Measures the randomness/variability of timing patterns across the entire session. Higher entropy indicates more natural, human-like timing.

---

## Feature Statistics

The feature extraction uses several statistical measures:

- **Mean**: Average value
- **Median**: Middle value (50th percentile)
- **Standard Deviation (std)**: Measure of variability
- **Skewness**: Measure of distribution asymmetry
- **Kurtosis**: Measure of distribution "tailedness"
- **Percentiles (p10, p25, p75, p90, p95, p99)**: Values below which a given percentage of observations fall
- **Entropy**: Measure of randomness/diversity in a distribution

---

## Why These Features Matter for Bot Detection

### Human Characteristics Captured:
1. **Natural Variability**: Humans show high variability in timing, speed, and movement patterns
2. **Micro-movements**: Natural hand tremor and micro-corrections
3. **Hesitation and Pauses**: Thinking time, reading pauses, and hesitation before actions
4. **Smooth Deceleration**: Natural slowing when approaching targets (Fitts' law)
5. **Bursty Patterns**: Activity in bursts with natural pauses
6. **Curved Paths**: Rarely perfectly straight movements
7. **Error Patterns**: Occasional mistakes and corrections

### Bot Characteristics Detected:
1. **Unnatural Regularity**: Too consistent timing and movement patterns
2. **Perfect Precision**: Lack of micro-movements and tremor
3. **Immediate Actions**: No hesitation or pauses
4. **Constant Speed**: Lack of natural acceleration/deceleration
5. **Straight Paths**: Overly direct movements
6. **Perfect Execution**: Lack of errors or corrections

---

## Feature Extraction Notes

- All timestamps are converted from milliseconds to seconds
- Missing features are filled with 0
- Infinite values are replaced with 0
- Features are extracted per session (a sequence of events for a user)
- The feature extraction handles edge cases (empty events, insufficient data, etc.)

---

## Total Feature Count

The system extracts **approximately 70+ features** across all categories, providing a comprehensive representation of user interaction patterns for effective bot detection.

