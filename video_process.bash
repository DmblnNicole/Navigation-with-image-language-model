ffmpeg -framerate 2 -pattern_type glob -i 'processed_hike/*.png'  -c:v libx264 -pix_fmt yuv420p processed_hike.mp4