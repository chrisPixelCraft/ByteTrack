#!/usr/bin/env python3

import os
import signal
import subprocess
import time

def find_tracking_processes():
    """Find all tracking-related processes"""
    try:
        # Find demo_track.py processes
        result = subprocess.run(['pgrep', '-f', 'demo_track.py'],
                              capture_output=True, text=True)
        demo_pids = result.stdout.strip().split('\n') if result.stdout.strip() else []

        # Find other tracking processes
        result = subprocess.run(['pgrep', '-f', 'python.*track'],
                              capture_output=True, text=True)
        track_pids = result.stdout.strip().split('\n') if result.stdout.strip() else []

        # Combine and deduplicate
        all_pids = list(set(demo_pids + track_pids))
        return [pid for pid in all_pids if pid and pid.isdigit()]

    except Exception as e:
        print(f"Error finding processes: {e}")
        return []

def get_process_info(pid):
    """Get detailed process information"""
    try:
        result = subprocess.run(['ps', '-p', pid, '-o', 'pid,ppid,cmd'],
                              capture_output=True, text=True)
        return result.stdout.strip().split('\n')[1] if result.returncode == 0 else None
    except:
        return None

def terminate_process(pid, force=False):
    """Terminate a process gracefully or forcefully"""
    try:
        if force:
            os.kill(int(pid), signal.SIGKILL)
            print(f"Force killed process {pid}")
        else:
            os.kill(int(pid), signal.SIGTERM)
            print(f"Sent SIGTERM to process {pid}")
        return True
    except ProcessLookupError:
        print(f"Process {pid} already terminated")
        return True
    except PermissionError:
        print(f"Permission denied to kill process {pid}")
        return False
    except Exception as e:
        print(f"Error killing process {pid}: {e}")
        return False

def cleanup_tracking_processes():
    """Main cleanup function"""
    print("🔍 Searching for tracking processes...")

    pids = find_tracking_processes()

    if not pids:
        print("✅ No tracking processes found!")
        return

    print(f"📋 Found {len(pids)} tracking process(es):")

    # Show process information
    for pid in pids:
        info = get_process_info(pid)
        if info:
            print(f"   PID {pid}: {info}")
        else:
            print(f"   PID {pid}: (process info unavailable)")

    # Ask for confirmation
    response = input(f"\n❓ Terminate {len(pids)} process(es)? [y/N]: ").lower()

    if response not in ['y', 'yes']:
        print("❌ Cleanup cancelled")
        return

    print("\n🛑 Terminating processes...")

    # First try graceful termination
    terminated = []
    for pid in pids:
        if terminate_process(pid, force=False):
            terminated.append(pid)

    # Wait a bit for graceful termination
    if terminated:
        print("⏳ Waiting 3 seconds for graceful termination...")
        time.sleep(3)

    # Check which processes are still running
    remaining_pids = find_tracking_processes()

    if remaining_pids:
        print(f"💀 Force killing {len(remaining_pids)} remaining process(es)...")
        for pid in remaining_pids:
            terminate_process(pid, force=True)

    # Final check
    final_pids = find_tracking_processes()

    if not final_pids:
        print("✅ All tracking processes terminated successfully!")
    else:
        print(f"⚠️  Warning: {len(final_pids)} process(es) still running:")
        for pid in final_pids:
            info = get_process_info(pid)
            print(f"   PID {pid}: {info}")

def cleanup_output_files():
    """Clean up incomplete output files"""
    import glob

    print("\n🧹 Cleaning up incomplete output files...")

    # Find incomplete tracking results (video files without txt files)
    track_vis_pattern = "YOLOX_outputs/yolox_m_mix_det/track_vis/2025_*"

    incomplete_dirs = []

    for dir_path in glob.glob(track_vis_pattern):
        if os.path.isdir(dir_path):
            timestamp = os.path.basename(dir_path)
            txt_file = f"YOLOX_outputs/yolox_m_mix_det/track_vis/{timestamp}.txt"

            # Check if there are video files but no txt file
            video_files = glob.glob(os.path.join(dir_path, "*.MP4"))

            if video_files and not os.path.exists(txt_file):
                incomplete_dirs.append(dir_path)

    if incomplete_dirs:
        print(f"📁 Found {len(incomplete_dirs)} incomplete tracking directories:")
        for dir_path in incomplete_dirs:
            print(f"   {dir_path}")

        response = input(f"\n❓ Remove {len(incomplete_dirs)} incomplete directories? [y/N]: ").lower()

        if response in ['y', 'yes']:
            import shutil
            for dir_path in incomplete_dirs:
                try:
                    shutil.rmtree(dir_path)
                    print(f"🗑️  Removed {dir_path}")
                except Exception as e:
                    print(f"❌ Failed to remove {dir_path}: {e}")
    else:
        print("✅ No incomplete tracking directories found")

if __name__ == "__main__":
    import sys

    print("🧹 ByteTrack Process Cleanup Tool")
    print("=" * 40)

    if len(sys.argv) > 1 and sys.argv[1] == "--files-only":
        cleanup_output_files()
    else:
        cleanup_tracking_processes()

        if input("\n❓ Also clean up incomplete output files? [y/N]: ").lower() in ['y', 'yes']:
            cleanup_output_files()

    print("\n🎉 Cleanup complete!")