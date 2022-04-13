from Inpainting import InpaintingModel
from Segmentation import SegmentationModel
from utilities import *
import cv2 as cv
import numpy as np
import time
import os

class Renderer():
    # Load inpainting and segmentation mpodels
    def __init__(self, lama_root):
        self.inpainter = InpaintingModel(lama_root)
        self.segmentation_model = SegmentationModel()

    # Render a given video
    def render_video(self, video_path, show_frames, save_frames, starting_frame=0, ending_frame=4294967295, bilateral_solve=True):
        # Video Capture
        cap = cv.VideoCapture(video_path)
        i = 0

        # Check for necessary file directories
        assert os.path.isdir('frames'), "No 'frames' folder found. Please create this folder."
        if save_frames:
            assert os.path.isdir('frames_color'), "No 'frames_color' folder found. Please create this folder."
            assert os.path.isdir('frames_inpainted'), "No 'frames_inpainted' folder found. Please create this folder."
            assert os.path.isdir('frames_seg'), "No 'frames_seg' folder found. Please create this folder."

        # Check if that starting frame is valid to the ending frame
        if starting_frame >= ending_frame:
            assert 1 == 0, "Starting frame must be before ending frame"

        while cap.isOpened():
            # Read the frame from the video capture
            r, frame = cap.read()

            # If the index is greater than the starting frame
            if r and i >= starting_frame and i < ending_frame:
                # Timing information
                start = time.time()

                # Write the frame
                cv.imwrite(f'frames/frame_{i}.png', frame)

                # Start frame analysis
                print(f'Analyzing Frame {i}')
                cv.imshow('color frame', frame)
                cv.waitKey(1)

                # Render a given frame
                self.render_frame(frame,
                                  kernel_size=473,
                                  seg_type='linear',
                                  frame_no=i,
                                  show_images=show_frames,
                                  save=save_frames,
                                  bilateral_solve=bilateral_solve)

                stop = time.time()
                print(f"\tFrame Rendering Time: {stop - start}s")

            elif not r:
                break

            i += 1

    # Render a specific frame with the methodology as listed
    def render_frame(self, frame, kernel_size, frame_no, seg_type = 'standard', show_images = True, save = True, bilateral_solve=True):
        # frame = cv.resize(frame, (frame.shape[1], frame.shape[0]))
        # frame = cv.pyrDown(frame, 1)
        i = frame_no

        # Analyze a given frame by deconstruction with either of these two methods
        assert seg_type == 'linear' or seg_type == 'standard' or seg_type == 'text_preprocessed'
        print(f'\tStarting Segmentation - {seg_type}')

        # Create binary segmentation mask of people from frame of video
        segmented_frame = np.zeros((frame.shape[0], frame.shape[1], 1))
        if seg_type == 'linear':
            segmented_frame = self.segmentation_model.average_linear_bucket(frame, kernel_size)
        if seg_type == 'standard':
            segmented_frame = self.segmentation_model.bucket_analyze(frame, kernel_size)
        if seg_type == 'text_preprocessed':
            segmented_frame = cv.imread(f'./frames_text_seg/frame_{frame_no}.png')
            segmented_frame = cv.resize(segmented_frame, (frame.shape[1], frame.shape[0]))[..., 0]
            segmented_frame = np.array([segmented_frame])
            segmented_frame = np.einsum('abc->bca', segmented_frame)

        # If there are people detected in the frame
        if not np.all((segmented_frame==0)):
            # Check if the Bilateral Solver flag is on
            if bilateral_solve == True:
                print(f'\tStarting Bilateral Solver')

                # Bilaterally Solve the segmentation with the frame to fit image
                segmented_frame = self.segmentation_model.bilateral_solve_segmentation(frame, segmented_frame)
            else:
                # Pad with a simple dilation step
                kernel = np.array([
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0],
                    [1, 0, 0, 0, 1],
                    [0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 0]
                ]).astype(np.uint8)     # Unique kernel works well - kind of interesting
                segmented_frame = cv.dilate(segmented_frame, kernel, iterations=6)

            # Combine segmentation with color image & save
            frame_comb = frame.copy()
            frame_comb[..., 0][segmented_frame >= 1] = 255
            # frame_comb[..., 1][segmented_frame >= 1] = 255
            # frame_comb[..., 2][segmented_frame >= 1] = 255

            # Get the inpainted frame
            print(f'\tInpainting...')
            inpainted = self.inpainter.predict_inpaint(frame, segmented_frame)

            # Show the images
            if show_images:
                cv.imshow('color segmentation', frame_comb)
                cv.imshow('segmentation', segmented_frame * 255)
                cv.imshow('inpainted', inpainted)
                cv.waitKey(1)

            # Save the images
            if save:
                cv.imwrite(f'./frames_seg/frame_{i}.png', segmented_frame * 255)
                cv.imwrite(f'./frames_color/frame_{i}.png', frame_comb)
                cv.imwrite(f'./frames_inpainted/frame_{i}.png', inpainted)

        # If no people are found in the frame
        else:
            inpainted = frame
            frame_comb = frame

            if show_images:
                cv.imshow('color segmentation', frame_comb)
                cv.imshow('segmentation', segmented_frame * 255)
                cv.imshow('inpainted', inpainted)
                cv.waitKey(1)

            if save:
                cv.imwrite(f'./frames_seg/frame_{i}.png', segmented_frame * 255)
                cv.imwrite(f'./frames_color/frame_{i}.png', frame_comb)
                cv.imwrite(f'./frames_inpainted/frame_{i}.png', inpainted)

        return frame_comb, inpainted

    # Run the median postprocessing model on the frame
    def median_postprocess(self, capture_path, num_steps_into_past):
        # MEDIAN POSTPROCESSING CODE
        past_frames = []
        i = 0

        assert os.path.isdir('frames_median_inpaint')

        cap = cv.VideoCapture(capture_path)
        assert cv.imread(f'frames_inpainted/frame_{i}.png') is not None, "Missing video render collection - run the Renderer to populate 'frames', 'frames_inpainted', and 'frames_seg' before running this code"

        while cap.isOpened():
            start = time.time()

            # Read the frame from the video capture
            r, frame = cap.read()
            inpainted_frame = cv.imread(f'frames_inpainted/frame_{i}.png')
            segmented_frame = (cv.imread(f'frames_seg/frame_{i}.png') / 255).astype(np.uint8)

            if not i < num_steps_into_past:
                past_frames.pop(0)

            if not r:
                break

            if i > 0:
                # Stack the previous several frames into one array
                median_frame = np.average(np.array(past_frames), axis=0).astype(np.uint8)
                inpainting_frame = median_frame * segmented_frame
                cv.imshow('inpainting_frame', inpainted_frame)
                past_frames.append(inpainted_frame)

                # Concatenate inpainted and non-inpainted parts to create final frame
                final_frame = np.zeros_like(inpainted_frame)
                print(inpainted_frame.shape)
                final_frame[segmented_frame == 0] = frame[segmented_frame == 0]
                final_frame[segmented_frame == 1] = inpainting_frame[segmented_frame == 1]

                # Timing Information
                end = time.time()
                print(f'\tFrame {i} : {round(end - start, 3)}s')

                # Save and display
                cv.imwrite(f'./frames_median_inpaint/frame_{i}.png', final_frame)
                cv.imshow('f', frame)
                show(final_frame, title='frame')
                show(segmented_frame * 255, title='seg')
                cv.waitKey(1)

            i += 1
        else:
            cap.release()
        pass

    # Load the precomputed text segmentation model
    def load_text_seg_masks(self):
        # Load the segmentation map from the transformer dataset
        pred_masks_per_frame = np.load('pred_masks_per_frame.npy')

        # For each frame in the array
        for i in range(pred_masks_per_frame.shape[0]):
            # Select the ith frame
            frame = pred_masks_per_frame[i,:,:,:]

            # Reshape the array dimensions
            frame = np.einsum('abc->bca', frame)

            # Write the frame to the appropriate folder
            cv.imwrite(f'./frames_text_seg/frame_{i}.png', frame)