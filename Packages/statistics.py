import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_flower_count(df):
  class_counts = df['Class'].value_counts()

  # Calculate the total number of flowers
  total_flowers = df.shape[0]

  # Calculate the number of unique classes
  unique_classes = df['Class'].nunique()

  # Display the statistics
  print("Class Counts:")
  print(class_counts)
  print("\nTotal Flowers:", total_flowers)
  print("Unique Classes:", unique_classes)

def get_class_distribution(df):
  class_counts = df['Class'].value_counts()
  class_percentage = df['Class'].value_counts(normalize=True) * 100

  plt.figure(figsize=(18, 5))
  plt.bar(class_counts.index, class_counts.values)
  plt.xlabel('Flower Class')
  plt.xticks(rotation=90)
  plt.ylabel('Count')
  plt.title('Flower Class Distribution')
  plt.show()

def get_stats(df, root):
  # Initialize an empty DataFrame to store image statistics
  df_stats = pd.DataFrame(columns=['Image', 'Width', 'Height', 'Channels', 'Mean_R', 'Mean_G', 'Mean_B', 'StdDev_R', 'StdDev_G', 'StdDev_B'])

  # Iterate over each image in the dataset
  for ID in df["ID"]:
      print(ID)
      # Read the image using OpenCV
      image_path = f"{root}/data/flowers-102/jpg/image_{ID}.jpg"
      image = cv2.imread(image_path)

      # Get image properties
      height, width, channels = image.shape

      # Calculate mean and standard deviation of pixel intensities
      b, g, r = cv2.split(image)

      # Calculate mean of each channel
      mean_r = np.mean(r)
      mean_g = np.mean(g)
      mean_b = np.mean(b)
      # mean = np.mean(image)
      std_dev_r = np.std(r)
      std_dev_g = np.std(g)
      std_dev_b = np.std(b)
      # std_dev = np.std(image)
      image_dict = {'Image': image_path,
                      'Width': width,
                      'Height': height,
                      'Channels': channels,
                      'Mean_R': mean_r,
                      'Mean_G': mean_g,
                      'Mean_B': mean_b,
                      'StdDev_R': std_dev_r,
                      'StdDev_G': std_dev_g,
                      'StdDev_B': std_dev_b}
      print(image_dict)
      # Append image statistics to DataFrame
      df_stats = df_stats.append(image_dict, ignore_index=True)

  print(df_stats.head())
  return df_stats


def plot_stats(df):
  # Calculate data statistics
  statistics = df.describe()

  # Plotting the data statistics
  plt.figure(figsize=(12, 8))

  # Histogram of Width
  plt.subplot(2, 3, 1)
  plt.hist(df['Width'], bins=20, edgecolor='black')
  plt.xlabel('Width')
  plt.ylabel('Frequency')
  plt.title('Histogram of Width')

  # Histogram of Height
  plt.subplot(2, 3, 2)
  plt.hist(df['Height'], bins=20, edgecolor='black')
  plt.xlabel('Height')
  plt.ylabel('Frequency')
  plt.title('Histogram of Height')
  plt.tight_layout()
  plt.show()

  # Box plots for channel-wise means
  plt.figure(figsize=(10, 5))
  plt.subplot(1, 2, 1)
  plt.boxplot([df['Mean_R'], df['Mean_G'], df['Mean_B']])
  plt.xticks([1, 2, 3], ['R', 'G', 'B'])
  plt.xlabel('Channel')
  plt.ylabel('Mean')
  plt.title('Box Plot of Channel-wise Means')

  # Box plots for channel-wise standard deviations
  plt.subplot(1, 2, 2)
  plt.boxplot([df['StdDev_R'], df['StdDev_G'], df['StdDev_B']])
  plt.xticks([1, 2, 3], ['R', 'G', 'B'])
  plt.xlabel('Channel')
  plt.ylabel('Standard Deviation')
  plt.title('Box Plot of Channel-wise Standard Deviations')


  plt.tight_layout()
  plt.show()

def show_image_classes(df):
  class_images = {}

  # Iterate over each row in the DataFrame
  for index, row in df.iterrows():
      image_path = row['ID']
      image_class = row['Class']

      # Check if the class already has an image assigned
      if image_class not in class_images:
          class_images[image_class] = image_path

  # Visualize one image of each class
  plt.figure(figsize=(10,50))

  for i, (class_name, image_path) in enumerate(class_images.items()):
      plt.subplot(25, 5, i+1)
      img = plt.imread(f"/content/drive/MyDrive/Fellowship AI/S11_superconvergence/data/flowers-102/jpg/image_{image_path}.jpg")
      plt.imshow(img)
      plt.title(class_name)
      plt.axis('off')

  plt.tight_layout()
  plt.show()

  return class_images
