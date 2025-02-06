import tensorflow as tf
import tensorflow_datasets as tfds

# Load only the test split
test_ds = tfds.load('kitti', split='test')

# Take a single example
for example in test_ds.take(1):
    image = example['image']
    
    # Convert to uint8 (if necessary for saving)
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    
    # Encode as JPEG
    encoded_jpeg = tf.io.encode_jpeg(image)
    
    # Write to file
    tf.io.write_file('test_image.jpg', encoded_jpeg)