export SEC=DA
export Name=DA_bus

ffmpeg -i outpaint_videos/$SEC/$Name.mp4 -vf "select='not(mod(n,2))'" -vsync vfr -pix_fmt yuv420p $Name.yuv
ffmpeg -i outpaint_videos/$SEC/$Name.mp4 -vf "select='not(mod(n,2))'" -vsync vfr -c:v libx264 -crf 0 $Name.mp4