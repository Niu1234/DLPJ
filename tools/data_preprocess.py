import os
import shutil
import subprocess
import sys

# Data copy
download_dir = "/content/drive/MyDrive/Data/Datasets/Anomaly-Detection-Dataset"
working_dir = "/content/drive/MyDrive/Data/Datasets/Anomaly-Detection-Dataset-mp4/Anomaly-Videos"

for filename in os.listdir(working_dir):
    file_path = os.path.join(working_dir, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

for i in range(1, 5):
    part_dir = os.path.join(download_dir, "Anomaly-Videos-Part-{}".format(i))
    for cls in os.listdir(part_dir):
        class_dir = os.path.join(part_dir, cls)
        subprocess.run(["cp", "-R", class_dir, working_dir], capture_output=True)

# MP4 to JPG
mp4_dir = "/content/drive/MyDrive/Data/Datasets/Anomaly-Detection-Dataset-mp4/Anomaly-Videos"
jpg_dir = "/content/drive/MyDrive/Data/Datasets/Anomaly-Detection-Dataset-jpg-fps30/Anomaly-Videos"
for i, cls in enumerate(sorted(os.listdir(mp4_dir))):
    if i == 0:
        mp4_class_dir = os.path.join(mp4_dir, cls)
        jpg_class_dir = os.path.join(jpg_dir, cls)

        commands = []
        for i, video_name in enumerate(os.listdir(mp4_class_dir)):
            video_path = os.path.join(mp4_class_dir, video_name)
            video_jpg_dir = os.path.join(jpg_class_dir, video_name[:-4])
            if not os.path.exists(video_jpg_dir):
                os.makedirs(video_jpg_dir)
            cmd = ['ffmpeg', '-i', video_path, '-vf', 'fps=30', '{}/%06d.jpg'.format(video_jpg_dir)]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            output, error = process.communicate()
            print(video_jpg_dir)
            print(output, error)
            commands.append(cmd)
