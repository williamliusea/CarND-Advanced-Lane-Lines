# Import everything needed to edit/save/watch video clips
import processimage as pimg
from moviepy.editor import VideoFileClip
import argparse
import matplotlib.image as mpimg
import camera
import numpy as np
import config

parser = argparse.ArgumentParser(description='Process single video image. \nLoad from test_videos/ and output to test_videos_output/.')
parser.add_argument('filename', type=str,
                   help='filename in test_videos directory')
# parser.add_argument('--speed', type=str, default='104',
                   # help='filename in test_videos directory')
args = parser.parse_args()

## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip1 = VideoFileClip(args.filename, audio=False, fps_source='fps').subclip(21,24)
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
image = clip1.get_frame(0.1)
img_size = (image.shape[1], image.shape[0])
cam = camera.Camera()
cam.load('camera.p')
src = np.float32(
    [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 25), img_size[1]],
    [(img_size[0] * 5 / 6) + 35, img_size[1]],
    [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
config = config.Config()
config.setPerspectiveMatrix(src, dst)
config.shape = image.shape
config.camera = cam
#config.y_per_frame = float(args.speed)/clip1.fps/config.ym_per_pix
#config.smooth_window = 720//config.y_per_frame
#print("y_per_frame:", config.y_per_frame, " smooth_window:", config.smooth_window)

processor = pimg.ProcessImage(config)

white_clip = clip1.fl_image(processor.process_image) #NOTE: this function expects color images!!
white_clip.write_videofile('test_videos_output/'+args.filename, audio=False)
print(processor.perf, processor.total)
