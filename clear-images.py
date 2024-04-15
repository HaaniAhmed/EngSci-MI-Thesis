from PIL import Image
import os

# Get a list of all the PNG files in the current folder
png_files = [file for file in os.listdir('.') if file.endswith('.png')]

# Loop through each PNG file and set the white background to clear
for file in png_files:
    # Open the image file
    img = Image.open(file)

    # Convert the image to RGBA mode
    img = img.convert('RGBA')

    # Get the pixel data for the image
    data = img.getdata()

    # Loop through each pixel and set the white background to clear
    newData = []
    for pixel in data:
        if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(pixel)

    # Update the pixel data for the image
    img.putdata(newData)

    # Save the modified image file
    img.save("images\\"+file)