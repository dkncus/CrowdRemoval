from utilities import *
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from keras_segmentation.models.all_models import model_from_name

class SegmentationModel():
    def __init__(self):
        self.model_segmentation = self.pspnet_50_ADE_20K()  # load the pretrained model trained on ADE20k dataset

    # Load the pretrained PSPNet Model
    def pspnet_50_ADE_20K(self):
        model_config = {
            "input_height": 473,
            "input_width": 473,
            "n_classes": 150,
            "model_class": "pspnet_50",
        }

        # Load a model from a given set of parameters and weights
        model = model_from_name[
            model_config['model_class']](
            model_config['n_classes'],
            input_height=model_config['input_height'],
            input_width=model_config['input_width']
        )

        # Load pretrained weights and return
        model.load_weights("pspnet50_ade20k.h5")

        return model

    # Average Linear Bucketing Algorithm - takes average line of best fit across frame
    def average_linear_bucket(self, frame, kernel_size):
        # Create empty array to store segmentation
        array = np.zeros((kernel_size, kernel_size, 1))
        frame_mod = frame.copy()

        # Create image segmentation from PSPNET_50 model
        out = self.model_segmentation.predict_segmentation(inp=frame).astype(np.uint8)

        # Segmentation
        # 115 = Bag/Handbag
        # 12 = Person
        # 32 = Also Person? Maybe?
        array[out == 12] = 1  # Specifically segment humans from the image
        array[out == 115] = 1  # Handbags
        # array[out == 32] = 1  # Persons?

        # Compute linear line of best fit from 473x473 frame with from positive hits on segmentation map
        y_x_coords = np.column_stack(np.where(array == 1))
        x = y_x_coords[:, 1]
        y = y_x_coords[:, 0]

        if x.size > 0:

            m, b = np.polyfit(x, y, 1)
            xs = np.array(range(0, kernel_size))
            line = m * xs + b

            # # Show a scatterplot of the information
            # plt.scatter(x, y)
            # plt.xlim(0, kernel_size)
            # plt.ylim(0, kernel_size)
            # plt.plot(xs, line, color='red')

            # Remap line of best fit to frame size
            sf_x = frame.shape[0] / kernel_size  # Scaling Factor (X Axis)
            sf_y = frame.shape[1] / kernel_size  # Scaling Factor (Y Axis)
            LOF_coords_1 = (int(xs[0] * sf_y), int(line[0] * sf_x))  # Find the starting point of line of fit
            LOF_coords_2 = (int(xs[-1] * sf_y), int(line[-1] * sf_x))  # Find the ending point of line of fit

            # Show the remapped line
            cv.line(frame_mod, LOF_coords_1, LOF_coords_2, (0, 255, 0), thickness=3)

            # Compute bucket centers along best fit line from bucket frequency
            sample_frequency = 6
            points = getEquidistantPoints(LOF_coords_1, LOF_coords_2, sample_frequency)

            # Select buckets for analysis from frame
            buckets = []
            buckets_location = []
            for j, point in enumerate(points):
                if not (j == 0 or j == sample_frequency):
                    point_int = [int(point[0]), int(point[1])]
                    cv.circle(frame_mod, point_int, radius=7, color=(0, 0, 255), thickness=-1)

                    # Adjust the point if the rectangle will fall out of bounds on the image
                    if point_int[0] < kernel_size // 2:
                        point_int[0] = kernel_size // 2 + 1
                    if point_int[1] < kernel_size // 2:
                        point_int[1] = kernel_size // 2 + 1

                    if point_int[0] > frame.shape[1] - kernel_size // 2:
                        point_int[0] = frame.shape[1] - (kernel_size // 2 + 1)
                    if point_int[1] > frame.shape[0] - kernel_size // 2:
                        point_int[1] = frame.shape[0] - (kernel_size // 2 + 1)

                    # Draw a rectangle based on the location of the selected image
                    rectangle_point_1 = (
                        (point_int[0] - (kernel_size // 2)) - 1,
                        (point_int[1] - (kernel_size // 2)) - 1
                    )
                    rectangle_point_2 = (
                        (point_int[0] + (kernel_size // 2 + 1)) - 1,
                        (point_int[1] + (kernel_size // 2 + 1)) - 1
                    )
                    rectangle = [rectangle_point_1, rectangle_point_2]
                    buckets_location.append(rectangle)

                    # Select the corresponding pixels from the frame
                    window = frame[
                             rectangle_point_1[1]:rectangle_point_2[1],
                             rectangle_point_1[0]:rectangle_point_2[0],
                             ]
                    buckets.append(window)
                    cv.rectangle(frame_mod, rectangle_point_1, rectangle_point_2,
                                 color=(255, 0, 0),
                                # color=(255 - 30 * j, 30 * j, 255 - 30 * j),
                                 thickness=5)

            cv.imshow('Segmentation - Average Linear Bucketing', frame_mod)

            # Predict each bucket
            p_buckets = self.analyze_linear_bucket_set(buckets)

            # Map per-bucket segmentation back onto 1920x1080 image (frames_seg)
            segmentation = np.zeros_like(frame)
            for j, bucket in enumerate(p_buckets):
                bucket_info = buckets_location[j]
                r_1 = bucket_info[0]
                r_2 = bucket_info[1]
                window = segmentation[r_1[1]:r_2[1], r_1[0]:r_2[0]]
                segmentation[r_1[1]:r_2[1], r_1[0]:r_2[0]] = np.logical_or(bucket, window)

            segmentation_1d = segmentation[..., 0]

            return segmentation_1d
        else:
            return np.zeros_like(frame[...,0])

    # Analyze buckets from linear average bucketing system
    def analyze_linear_bucket_set(self, buckets):
        # Create empty set of buckets
        p_buckets = []

        # For each bucket along the line
        for bucket in buckets:
            # Create empty array to store segmentation
            array = np.zeros((bucket.shape[0], bucket.shape[1], 1))

            # Create image segmentation from PSPNET_50 model
            out = self.model_segmentation.predict_segmentation(inp=bucket).astype(np.uint8)
            array[out == 12] = 1  # Specifically segment humans from the image
            array[out == 115] = 1  # Add Human handbags to the image
            p_buckets.append(array)  # Append to the row

        return p_buckets

    # Collect each individual window from the frame
    def collect_prediction_sets(self, frame, kernel_size):
        # All collected predictions
        core_prediction_set = []
        vertical_edges = []
        horizontal_edges = []

        # Get the height and width of the image
        height = frame.shape[0]
        width = frame.shape[1]
        height_windows = height // kernel_size  # Number of height windows
        width_windows = width // kernel_size    # Number of width windows

        ### Core image extraction ###
        for n_x in range(height_windows):
            # Create empty array to store the row of windows
            row = []

            for n_y in range(width_windows):
                # Slice the given window from the frame
                window = frame[
                            n_x * kernel_size:(n_x + 1) * kernel_size,
                            n_y * kernel_size:(n_y + 1) * kernel_size
                         ]

                # Append the sliced window to the selection set
                row.append(window)

            # Append the row slices to the prediction set
            core_prediction_set.append(row)

        ### Edge Cases (no pun intended) ###

        # Vertical Edge Case
        for n in range(height_windows):
            # Slice vertical edge window from the frame
            window = frame[
                        n * kernel_size : (n + 1) * kernel_size,
                        width - kernel_size : width
                    ]

            # Append to set of vertical edges
            vertical_edges.append(window)

        # Horizontal edge cases
        for n in range(width_windows):
            # Slice vertical edge window from the frame
            window = frame[
                        height - kernel_size: height,
                        n * kernel_size : (n + 1) * kernel_size
                     ]

            # Append to set of vertical edges
            horizontal_edges.append(window)

        # Bottom Right Corner Case
        corner_edge = frame[
                 height - kernel_size: height,
                 width - kernel_size : width
        ]

        return core_prediction_set, vertical_edges, horizontal_edges, corner_edge

    # Analyze each individual sample through the network
    def analyze_sets(self, core, vertical, horizontal, corner):
        p_core = []
        p_vertical = []
        p_horizontal = []

        # For each row in the core image
        for row in core:

            p_row = []
            for image in row:
                # Create empty array to store segmentation
                array = np.zeros((image.shape[0], image.shape[1], 1))

                # Create image segmentation from PSPNET_50 model
                out = self.model_segmentation.predict_segmentation(inp=image).astype(np.uint8)
                array[out == 12] = 1    # Specifically segment humans from the image
                array[out == 115] = 1    # Add Human handbags to the image
                p_row.append(array)     # Append to the row

            # Append each row to the core
            p_core.append(p_row)

        # For each image in the vertical edges
        for image in vertical:
            # Create empty array to store segmentation
            array = np.zeros((image.shape[0], image.shape[1], 1))

            # Create image segmentation from PSPNET_50 model
            out = self.model_segmentation.predict_segmentation(inp=image).astype(np.uint8)
            array[out == 12] = 1      # Specifically segment humans from the image
            array[out == 115] = 1     # Add Human handbags to the image
            p_vertical.append(array)  # Append to the row

        # For each image in the horizontal edges
        for image in horizontal:
            # Create empty array to store segmentation
            array = np.zeros((image.shape[0], image.shape[1], 1))

            # Create image segmentation from PSPNET_50 model
            out = self.model_segmentation.predict_segmentation(inp=image).astype(np.uint8)
            array[out == 12] = 1  # Specifically segment humans from the image
            array[out == 115] = 1  # Add Human handbags to the image
            p_horizontal.append(array)  # Append to the row

        # Predict from the bottom right corner image
        array = np.zeros((corner.shape[0], corner.shape[1], 1))
        out = self.model_segmentation.predict_segmentation(inp=corner).astype(np.uint8)
        array[out == 12] = 1  # Specifically segment humans from the image
        array[out == 115] = 1  # Add Human handbags to the image
        p_corner = array

        return p_core, p_vertical, p_horizontal, p_corner

    # Stitch image components together
    def stitch_image(self, frame, kernel_size, core, vertical, horizontal, corner):
        # Get the height and width of the image
        height = frame.shape[0]
        width = frame.shape[1]
        height_windows = height // kernel_size  # Number of height windows
        width_windows = width // kernel_size  # Number of width windows

        # Calculate the number of pixels padding the border which need to be accounted for
        border_px_vertical = width - width_windows * kernel_size
        border_px_horizontal = height - height_windows * kernel_size

        rows = []

        # Stitch together the images in the core and on horizontal axes
        for i, row in enumerate(core):
            # Get the vertical slice corresponding to the given row
            slice = vertical[i][:,kernel_size-border_px_vertical:kernel_size]

            # Append the slice to the row, concatenate, and append to the list of rows
            row.append(slice)
            row_full = cv.hconcat(row)
            rows.append(row_full)

        final_row = []

        # For each image in the set of horizontal padding images
        for image in horizontal:
            # Take a horizontal slice from the border padding and append
            slice = image[kernel_size-border_px_horizontal:kernel_size, :]
            final_row.append(slice)

        # Concatenate the final row with the corner border pixels
        final_row = cv.hconcat(final_row)
        final_row = cv.hconcat([final_row, corner[kernel_size-border_px_horizontal:kernel_size, kernel_size-border_px_vertical:kernel_size]])

        # Concatenate the final image
        partial_image = cv.vconcat(rows)
        segmented_frame = cv.vconcat([partial_image, final_row])

        return segmented_frame

    # Bucketing analysis of the image (Analyzes 15 total images)
    def bucket_analyze(self, frame, kernel_size):
        core, vertical, horizontal, corner = self.collect_prediction_sets(frame, kernel_size)
        p_core, p_vertical, p_horizontal, p_corner = self.analyze_sets(core, vertical, horizontal, corner)
        segmented_frame = self.stitch_image(frame, kernel_size, p_core, p_vertical, p_horizontal, p_corner)

        return segmented_frame

    # Employ a bilateral solver to refine segmentation
    def bilateral_solve_segmentation(self, frame, segmented_frame):
        from FastBilateralSolver import FastBilateralSolver

        # Dilate segmented frame to remove unwanted edge pixels
        fbls = FastBilateralSolver(reference=frame,
                                   target=segmented_frame,
                                   confidence=np.ones_like(segmented_frame),
                                   luma=32,
                                   chroma=4,
                                   spatial=4
                                   )
        output = fbls.solve()
        output = output * 65535
        output = cv.GaussianBlur(output, ksize=(33, 33), sigmaX=4)

        output[output > 0.1] = 1
        output[output <= 0.1] = 0

        dilate_kernel = np.ones((3, 3), np.uint8)
        output = cv.dilate(output, dilate_kernel, iterations=4)

        output = output.astype(np.uint8)
        # show(output * 255, title=f'output')

        return output