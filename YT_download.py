import yt_dlp

url = "https://www.youtube.com/watch?v=-fYanG5hEpI&list=PLX6HSbSIHpCbcjxQF5P8fqX5hu2nxE4rx&index=6"

options = {
    "format": "bestvideo+bestaudio/best",
    "outtmpl": "%(title)s.%(ext)s",
    "merge_output_format": "mp4",
    "noplaylist": True,
}

with yt_dlp.YoutubeDL(options) as ydl:
    ydl.download([url])