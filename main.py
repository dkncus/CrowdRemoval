from Renderer import Renderer
import os
import cv2 as cv

if __name__ == '__main__':
    # Root directory of the Large Mask Inpainting algorithm & video path to render
    lama_root = './lama'
    video_path = 'sample_videos/trinity_clipped.mp4'

    # Check that both of these exist before proceeding
    assert os.path.exists(video_path), "The video specified does not exist in the root directory."
    assert os.path.exists(lama_root), "Provided LaMa model directory does not exist."

    # Create a renderer object and process video
    renderer = Renderer(lama_root)
    renderer.render_video(video_path, starting_frame=168, show_frames=True, save_frames=True, bilateral_solve=False)   # Segment and Inpaint each frame individually
    renderer.median_postprocess(video_path, num_steps_into_past=5)                                                     # Perform Median Postprocessing step
    # renderer.compile_video()

    # Single image processing
    # frame = cv.imread('input.png')
    # image = renderer.render_frame(frame, 473, 0, seg_type = 'linear')
    # cv.imwrite('output.png', image)