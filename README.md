# Human Crowd Removal in Non-Stationary Camera Video by Semantic Segmentation and Neural Inpainting
David Kubala, Trinity College Dublin

In recent years, there has been revolutionary re- search in the field of video object removal. There have been particularly notable advancements in two areas: semantic segmentation - the classification of image pixels into one of a number of subclasses, and inpainting - the interpretation and replace- ment of missing data from images. Given these two advancements, this paper proposes a new tech- nique for combining the strengths of both to per- form the automatic removal of people from crowded areas in images from both non-stationary camera video and single-frame captures.

# How to run

1. Create the following folders in root directory:
- frames
- frames_color
- frames_inpainted
- frames_median_inpaint
- frames_seg
- frames_text_seg

2. Download the PSPNet Segmentation model from <INSERT GOOGLE DRIVE LINK> and place in the root directory

3. Download the LaMa model from <INSERT GOOGLE DRIVE LINK> and place in the root directory
  
4. Add your own sample video in .mp4 format (preferably 720p) to the root directory, or download a sample from <INSERT GOOGLE DRIVE LINK>
  
5. Replace the string after 'video_path' in main.py with the path to your sample video

6. Run main.py
