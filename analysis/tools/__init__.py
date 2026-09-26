"""Standalone viewers and helpers; none of them is part of the pipeline.

    plot_imu            Dash dashboard of the raw accelerometer/gyroscope channels
    plot_kinematics     the kinematics check's sensor table and joint-angle panels, outside the dashboard
    animate_kinematics  3D stick-figure animation from the orientation cache (needs ffmpeg)
    plot_frame          one skeleton frame with local axis triads, to check sensor alignment
    method_checks       re-runs the checks the method choices rest on, and prints their numbers
    boundary_robustness jitters every event boundary and reports each metric's ICC
    yt_download         download the video at the URL in the script (needs yt-dlp)
"""
