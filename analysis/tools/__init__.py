"""Standalone viewers and helpers; none of them is part of the pipeline.

    plot_imu            Dash dashboard of the raw accelerometer/gyroscope channels
    plot_kinematics     the 50 Hz joint-angle series, in groups
    animate_kinematics  3D stick-figure animation from the orientation cache (needs ffmpeg)
    plot_frame          one skeleton frame with local axis triads, to check sensor alignment
    skeleton            the stick-figure definition the last two share
    yt_download         download the video at the URL in the script (needs yt-dlp)
"""
