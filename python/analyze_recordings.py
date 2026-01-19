"""
Analyze and validate PuckCapture recordings.

Usage:
    python analyze_recordings.py path/to/recordings
    python analyze_recordings.py path/to/recordings --verbose
"""

import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from load_recordings import load_episode, EpisodeMetadata

OUTCOMES = {0: "Goal", 1: "Timeout", 2: "Manual"}
CAPTURE_RATE_HZ = 50  # Recording rate


def detect_inactivity(actions: np.ndarray, threshold: float = 0.05) -> list:
    """
    Detect periods of player inactivity.

    Inactivity = movement input near zero (player not pressing WASD/stick).
    Note: Player may still be moving due to momentum.

    Returns list of (start_step, end_step, duration_seconds) tuples.
    """
    # Movement is cols 0-1
    movement = actions[:, :2]

    # Check if movement magnitude is below threshold
    movement_magnitude = np.sqrt(movement[:, 0]**2 + movement[:, 1]**2)

    inactive = movement_magnitude < threshold

    # Find consecutive inactive periods
    periods = []
    in_period = False
    start = 0

    for i, is_inactive in enumerate(inactive):
        if is_inactive and not in_period:
            start = i
            in_period = True
        elif not is_inactive and in_period:
            duration = (i - start) / CAPTURE_RATE_HZ
            periods.append((start, i, duration))
            in_period = False

    # Handle period at end
    if in_period:
        duration = (len(inactive) - start) / CAPTURE_RATE_HZ
        periods.append((start, len(inactive), duration))

    return periods


def get_inactivity_stats(actions: np.ndarray, min_duration: float = 1.0) -> dict:
    """Get inactivity statistics for an episode."""
    periods = detect_inactivity(actions)

    # Filter to significant periods (> min_duration seconds)
    significant = [p for p in periods if p[2] >= min_duration]

    total_inactive = sum(p[2] for p in significant)
    longest = max((p[2] for p in significant), default=0)

    return {
        "periods": significant,
        "count": len(significant),
        "total_seconds": total_inactive,
        "longest_seconds": longest,
    }


def analyze_recordings(folder: str, verbose: bool = False):
    """Analyze all recordings in a folder."""
    folder = Path(folder)
    files = list(folder.glob("*.bin"))

    if not files:
        print(f"No .bin files found in {folder}")
        return

    print(f"Found {len(files)} recording files\n")

    # Collect stats
    episodes_by_outcome = defaultdict(list)
    episodes_by_player = defaultdict(list)
    all_episodes = []
    errors = []

    for filepath in sorted(files):
        try:
            obs, actions, rewards, meta = load_episode(filepath)
            all_episodes.append((obs, actions, rewards, meta))
            episodes_by_outcome[meta.outcome].append(meta)
            episodes_by_player[meta.steam_id].append(meta)

            if verbose:
                print(f"  {filepath.name}")
                print(f"    Steps: {meta.step_count}, Duration: {meta.duration:.1f}s, Outcome: {OUTCOMES.get(meta.outcome, 'Unknown')}")
                print(f"    Player: {meta.username} ({meta.steam_id})")
                print(f"    Team: {'Blue' if meta.team == 0 else 'Red'}, Hand: {'Right' if meta.handedness == 0 else 'Left'}")
                print()
        except Exception as e:
            errors.append((filepath.name, str(e)))
            if verbose:
                print(f"  ERROR: {filepath.name} - {e}\n")

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nTotal episodes: {len(all_episodes)}")
    if errors:
        print(f"Failed to load: {len(errors)}")

    # Outcome breakdown
    print("\n--- By Outcome ---")
    for outcome_id in sorted(episodes_by_outcome.keys()):
        episodes = episodes_by_outcome[outcome_id]
        outcome_name = OUTCOMES.get(outcome_id, f"Unknown({outcome_id})")
        durations = [e.duration for e in episodes]
        steps = [e.step_count for e in episodes]

        print(f"\n{outcome_name}: {len(episodes)} episodes")
        if durations:
            print(f"  Duration: {np.mean(durations):.1f}s avg, {np.min(durations):.1f}s min, {np.max(durations):.1f}s max")
            print(f"  Steps: {np.mean(steps):.0f} avg, {np.min(steps)} min, {np.max(steps)} max")

    # Goal-specific stats (the main metric)
    goal_episodes = episodes_by_outcome.get(0, [])
    if goal_episodes:
        goal_durations = [e.duration for e in goal_episodes]
        print("\n" + "=" * 60)
        print("GOAL EPISODES (successful demonstrations)")
        print("=" * 60)
        print(f"  Count: {len(goal_episodes)}")
        print(f"  Average time to score: {np.mean(goal_durations):.2f} seconds")
        print(f"  Median time to score: {np.median(goal_durations):.2f} seconds")
        print(f"  Fastest: {np.min(goal_durations):.2f} seconds")
        print(f"  Slowest: {np.max(goal_durations):.2f} seconds")
        print(f"  Std dev: {np.std(goal_durations):.2f} seconds")

    # Player breakdown
    print("\n--- By Player ---")
    for steam_id, episodes in sorted(episodes_by_player.items(), key=lambda x: -len(x[1])):
        username = episodes[0].username
        goals = sum(1 for e in episodes if e.outcome == 0)
        total_duration = sum(e.duration for e in episodes)
        print(f"  {username} ({steam_id}): {len(episodes)} episodes, {goals} goals, {total_duration/60:.1f} min recorded")

    # Inactivity analysis
    print("\n" + "=" * 60)
    print("INACTIVITY ANALYSIS")
    print("=" * 60)

    episodes_with_inactivity = []
    for obs, actions, rewards, meta in all_episodes:
        stats = get_inactivity_stats(actions, min_duration=3.0)  # 3+ second gaps
        if stats["count"] > 0:
            episodes_with_inactivity.append((meta, stats))

    if episodes_with_inactivity:
        print(f"\nFound {len(episodes_with_inactivity)} episodes with inactivity (3+ sec gaps):\n")
        # Sort by longest inactive period
        episodes_with_inactivity.sort(key=lambda x: -x[1]["longest_seconds"])

        for meta, stats in episodes_with_inactivity[:10]:  # Show top 10
            pct = (stats["total_seconds"] / meta.duration) * 100
            print(f"  {meta.filename}")
            print(f"    Duration: {meta.duration:.1f}s, Inactive: {stats['total_seconds']:.1f}s ({pct:.0f}%)")
            print(f"    Longest gap: {stats['longest_seconds']:.1f}s, Gaps: {stats['count']}")

        if len(episodes_with_inactivity) > 10:
            print(f"\n  ... and {len(episodes_with_inactivity) - 10} more")
    else:
        print("\nNo significant inactivity periods detected (all episodes have continuous input)")

    # Data validation
    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    issues = []

    # Flag high-inactivity episodes
    for meta, stats in episodes_with_inactivity:
        pct = (stats["total_seconds"] / meta.duration) * 100
        if pct > 30:
            issues.append(f"{meta.filename}: {pct:.0f}% inactive time")

    # Check team/handedness consistency
    teams = set(e[3].team for e in all_episodes)
    hands = set(e[3].handedness for e in all_episodes)
    if len(teams) > 1:
        issues.append(f"Mixed teams found: {teams} (expected all 0=Blue)")
    if len(hands) > 1:
        issues.append(f"Mixed handedness found: {hands} (expected all 0=Right)")

    # Check observation ranges
    if all_episodes:
        all_obs = np.concatenate([e[0] for e in all_episodes], axis=0)
        all_actions = np.concatenate([e[1] for e in all_episodes], axis=0)

        print(f"\nObservation stats ({all_obs.shape[1]} features):")
        print(f"  Shape: {all_obs.shape}")
        print(f"  Range: [{all_obs.min():.3f}, {all_obs.max():.3f}]")
        print(f"  Mean: {all_obs.mean():.3f}")
        print(f"  Has NaN: {np.any(np.isnan(all_obs))}")
        print(f"  Has Inf: {np.any(np.isinf(all_obs))}")

        print(f"\nAction stats (8 features):")
        print(f"  Shape: {all_actions.shape}")
        print(f"  Range: [{all_actions.min():.3f}, {all_actions.max():.3f}]")

        # Check for reasonable action ranges
        move_actions = all_actions[:, :2]  # move_x, move_y
        aim_actions = all_actions[:, 2:4]  # aim_x, aim_y

        print(f"\n  Movement (cols 0-1): [{move_actions.min():.3f}, {move_actions.max():.3f}]")
        print(f"  Aim (cols 2-3): [{aim_actions.min():.3f}, {aim_actions.max():.3f}]")
        print(f"  Blade angle (col 4): [{all_actions[:, 4].min():.3f}, {all_actions[:, 4].max():.3f}]")
        print(f"  Buttons (cols 5-7): [{all_actions[:, 5:].min():.3f}, {all_actions[:, 5:].max():.3f}]")

        # Check for NaN/Inf
        if np.any(np.isnan(all_obs)):
            issues.append("Observations contain NaN values!")
        if np.any(np.isinf(all_obs)):
            issues.append("Observations contain Inf values!")
        if np.any(np.isnan(all_actions)):
            issues.append("Actions contain NaN values!")

    # Report issues
    if issues:
        print("\n[!] ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n[OK] No issues found - data looks good!")

    if errors:
        print(f"\n[!] Failed to load {len(errors)} files:")
        for name, err in errors:
            print(f"  - {name}: {err}")

    return all_episodes


def main():
    parser = argparse.ArgumentParser(description="Analyze PuckCapture recordings")
    parser.add_argument("folder", help="Path to recordings folder")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show details for each file")
    args = parser.parse_args()

    analyze_recordings(args.folder, args.verbose)


if __name__ == "__main__":
    main()
