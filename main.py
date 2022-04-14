from Renderer import Renderer
import os
import cv2 as cv

# Check that everything is initialized
def check_init():
    # Filepaths to check for existence
    paths = ['frames', 'frames_color', 'frames_inpainted', 'frames_median_inpaint', 'frames_seg', 'frames_text_seg']

    # For each of the necessary paths
    for path in paths:
        # If it doesn't exist
        if not os.path.exists(path):
            # Make it exist
            os.mkdir(path)

    assert os.path.isfile('pspnet50_ade20k.h5'), "The segmentation model weights are missing. Download them from https://bit.ly/3JHWZy7"
    assert os.path.exists('lama'), "The inpainting model is missing. Download it from <INSERT DRIVE LINK>"

def main():
    # Check that the initialization conditions are met
    check_init()

    # Video path to render
    video_path = 'input.mp4'

    # Check that both of these exist before proceeding
    assert os.path.exists(video_path), "The video specified does not exist in the root directory."
    assert os.path.exists('./lama'), "Provided LaMa model directory does not exist."

    # Create a renderer object and process video
    renderer = Renderer('./lama')

    # Segment and Inpaint each frame individually
    renderer.render_video(video_path,
                          starting_frame=0,
                          show_frames=True,
                          save_frames=True,
                          bilateral_solve=False)

    # Perform Median Postprocessing step
    renderer.median_postprocess(video_path, num_steps_into_past=5)
    # renderer.compile_video()

    # Single image processing
    # frame = cv.imread('input.png')
    # image = renderer.render_frame(frame, 473, 0, seg_type = 'linear')
    # cv.imwrite('output.png', image)


if __name__ == '__main__':
    main()