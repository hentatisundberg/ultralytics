import os


video_dir = '/home/shreyash_kad/fish_detection/dataset/test_video'
project_dir = 'runs/detect/fish_for_annotations'
shell_script = video_dir.split('/')[-1] + '.sh'
cmds = []
for video in os.listdir(video_dir):
    video_path = os.path.join(video_dir, video)
    print(video, '-------------------------------------------------------------')
    cmds.append(f"python detect.py --weights runs/train/new_annotations_X_1280sz/weights/best.pt --source {video_path} --data data/fish_data.yaml --view-img --imgsz 1280 --device 1 --project {project_dir} --name {video[:-4]} --for_annotations --save-txt")

if os.path.isfile(shell_script):
    os.remove(shell_script)

with open(shell_script, 'x') as f:
    for line in cmds:
        f.write(f"{line}\n")
#python detect.py --weights runs/train/new_annotations_X_1280sz/weights/best.pt --source test.mp4 --data data/fish_data.yaml --imgsz 1280 --device 1 --name FAR3_2023_07_01_041200
