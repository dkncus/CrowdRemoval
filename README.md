# Human Crowd Removal in Non-Stationary Camera Video by Semantic Segmentation and Neural Inpainting
David Kubala, Trinity College Dublin

In recent years, there has been revolutionary research in the field of object removal. There have been particularly notable advancements in two areas: semantic segmentation - the classification of image pixels into one of a number of subclasses, and inpainting - the interpretation and replacement of missing data from images. Given these two advancements, this paper proposes a new technique for combining the strengths of both to perform the automatic removal of people from crowded areas in images from both non-stationary camera video and single-frame captures.

![Thank you for reading my alt text! <3](https://github.com/dkncus/CrowdRemoval/blob/master/images/cover_photo.png)

# How to run

1. Install the required packages by the command line at the root directory with

`pip install -r requirements.txt`

3. Download the PSPNet Segmentation model from https://bit.ly/3JHWZy7 and place in the root directory

4. Download the LaMa model from <INSERT GOOGLE DRIVE LINK> and place in the root directory
  
5. Add your own sample video in .mp4 format (preferably 720p) to the root directory, or download a sample from <INSERT GOOGLE DRIVE LINK>
  
6. Replace the string after 'video_path' in main.py with the path to your sample video

7. Run main.py

  
![Check the alt text on the next image for a special suprise!](https://github.com/dkncus/CrowdRemoval/blob/master/images/portfolio_1.jpg)
                                                                                                                    
![No, not this one. The next one :D](https://github.com/dkncus/CrowdRemoval/blob/master/images/portfolio_2.jpg)
  
![Ȯ̸̱̦͚̜͗͌̏̏̇̏̏̓̈́̒̓̃̑͘Ḩ̶̙̱̰͉̯̭̬́́̿̃͑̄͗̓͘͝ ̵̮̠̰̹͇̳͕̄̆̉̚̚͜G̵͕̯̘̹̣̮̜̩̠̹̋O̷̤̼͉̭͈͖̒̊̇͗͑͛͗͛̅̔̚͝͝͝D̴̫̣̋̃̏̇̈́͠͠͝͠ ̶̟͛̈́̇̏͒̑̉̀̊́̋͠N̶̳͛́̂̆̎̔̍͗̈́̊̀͑͋̍̚Ǒ̷̊́ͅȚ̶̂̉̍́̑̌ ̵͉̯̜̖̝̔̽̾̉̿̽̑̽̍̈́T̸͇̺̜͍̩̜̥͇͇̈H̵̡͔͓̱̥͖͈̺͚̤͔͍͚͈͊̈́̒̄̈́̕Ị̵̛͖̞̯͓̊̌̎͑͑̾̕̕S̴̨̡̨̛͔̞͕͈̟͈̈͌͛̎̒̀̈́͜͝ ̷̡̡̯̳͙͍̣̣̤̜̔͐̍̋̇Ô̴̻͚͔͔͒͊̎̄͠ͅŅ̵͔̹̳̱̝̝̘͇̠̟̘̯̣̻̈̐̓̈͂̓͐͠͠Ȩ̴̧̭̩̖̥̈](https://github.com/dkncus/CrowdRemoval/blob/master/images/portfolio_3.jpg)
