import tensorflow.io
import tensorflow_datasets


def save_plain_dataset(dataset, path, dataset_type):

    # load the dataset
    dataset = tensorflow_datasets.load(dataset, split=dataset_type)

    # iterate over the dataset and save the images
    for data in dataset:
        image = data['image']
        filename = data['image/file_name']

        image_path = path + filename

        encoded_image = tensorflow.image.encode_png(image)

        tensorflow.io.write_file(image_path, encoded_image)

        print(f"Saved {image_path}")

def save_dataset(dataset, path, dataset_type):

    # load the dataset
    dataset = tensorflow_datasets.load(dataset, split=dataset_type)

    # define colors
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    # iterate over the dataset and save the image with bbox
    for data in dataset:
        image = data['image']
        bbox = data['objects']['bbox']
        filename = data['image/file_name']

        image_path = path + filename

        image = tensorflow.image.convert_image_dtype(image, dtype=tensorflow.float32)

        # current format of bbox is (ymin, xmin, ymax, xmax)
        # currently (0,0) is at the bottom-left
        # set (0,0) to top-left
        bbox = tensorflow.stack([
            1 - bbox[:, 2],
            bbox[:, 1],
            1 - bbox[:, 0],
            bbox[:, 3]
        ], axis=-1)

        # draw the bounding box on the image
        image = tensorflow.image.draw_bounding_boxes(
            tensorflow.expand_dims(image, 0),
            tensorflow.expand_dims(bbox, 0),
            colors
        )

        image = tensorflow.squeeze(image, 0)

        encoded_image = tensorflow.image.encode_png(tensorflow.image.convert_image_dtype(image, dtype=tensorflow.uint8))

        tensorflow.io.write_file(image_path, encoded_image)

        print(f"Saved {image_path}")

if __name__ == '__main__':
    # save_plain_dataset('kitti', '../../data/kitti-test-plain/', 'test')
    save_dataset('kitti', '../../data/kitti-test/', 'test')