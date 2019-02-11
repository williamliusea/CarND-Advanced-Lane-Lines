from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
import argparse

parser = argparse.ArgumentParser(description='Get a frame from video. \nLoad from test_videos/')
parser.add_argument('filename', type=str,
                   help='filename in test_videos directory')
parser.add_argument('timestamp', type=float,
                   help='extra the frame at this timestamp')
parser.add_argument('output_filename', type=str,
                   help='output the image to this file')
args = parser.parse_args()

##### white_output = 'test_videos_output/night-highway-clearlane-bridge-curve.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip1 = VideoFileClip(args.filename)
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
image = clip1.get_frame(args.timestamp)
print('This image is:', type(image), 'with dimensions:', image.shape)
# plt.imshow(result)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
mpimg.imsave(args.output_filename, image)
