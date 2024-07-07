# resize all images in a given folder to a given width

import os
import cv2
import argparse
from PIL import Image, ExifTags
from tqdm import tqdm

# Parse arguments
parser = argparse.ArgumentParser(description='Resize all images in a folder to a given width')

parser.add_argument('folder', type=str, help='Path to the folder containing the images')
parser.add_argument('width', type=int, help='Width to resize the images to', default=720)

args = parser.parse_args()

new_width = args.width
folder = args.folder

resize_progress = tqdm(total=len(os.listdir(args.folder)), desc='Resizing images')

print('Resizing images in folder', args.folder, 'to width', new_width)
except_count = 0
for filename in os.listdir(args.folder):
    try:
        img = Image.open(os.path.join(args.folder, filename)).convert('RGB')
    except:
        os.remove(os.path.join(args.folder, filename))
        except_count += 1
        continue
    # Correct orientation
    try:
        for orientation in ExifTags.TAGS.keys() : 
            if ExifTags.TAGS[orientation]=='Orientation' : break 
        
        exif=dict(img._getexif().items())

        if   exif[orientation] == 3 : 
            img=img.rotate(180, expand=True)
        elif exif[orientation] == 6 : 
            img=img.rotate(270, expand=True)
        elif exif[orientation] == 8 : 
            img=img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass
    
    # Resize
    aspect_ratio = img.height / img.width
    new_height = int(new_width * aspect_ratio)
    img = img.resize((new_width, new_height))
    img.save(folder + filename)
    resize_progress.update(1)

print('Resized', len(os.listdir(args.folder)) - except_count, 'images')
print('Removed', except_count, 'images')
resize_progress.close()