import os
import csv
print(os.getcwd())
# Set the directory where your images are stored
image_dir = '/mnt/FEF82D4EF82D070D/Studies/MSc/Masters-Project/Project/Model1/validation/male'

# Set the name of the output CSV file
output_file = '/mnt/FEF82D4EF82D070D/Studies/MSc/Masters-Project/Project/Model1/validation/male/labels.csv'

# Set the text you want to use as the label
mytext = 'male'

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

# Open the output CSV file for writing
with open(output_file, 'w') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write the header row
    writer.writerow(['filename', 'label'])
    
    # Write a row for each image file
    for image_file in image_files:
        writer.writerow([image_file, mytext])
