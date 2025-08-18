import os
import re
import yt_dlp
import pandas as pd

# Read CSV file
csv_path = 'go_dataset.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Extract all YouTube URLs using regex
youtube_links = re.findall(r'https://www\.youtube\.com/watch\?v=[\w\-]+', content)
print(f'Total {len(youtube_links)} URLs discovered')

# Set download folder
output_dir = 'go_videos'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# yt-dlp download configuration (720p mp4 video-only)
ydl_opts = {
    'format': 'bestvideo[ext=mp4][height<=720]',
    'outtmpl': os.path.join(output_dir, '%(title)s_%(height)sp.%(ext)s'),
    'noplaylist': True,
    'quiet': False,
    'no_warnings': True
}

for idx, url in enumerate(youtube_links, start=1):
    print(f'\n[{idx}/{len(youtube_links)}] Downloading: {url}')
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"Download failed: {url}, Reason: {e}")

print('All downloads completed!')