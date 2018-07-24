import random
import tensorflow as tf
from dataset_utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset, write_file_pattern_file
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#====================================================DEFINE YOUR ARGUMENTS=======================================================================
flags = tf.app.flags

#State your dataset directory
flags.DEFINE_string('dataset_dir', None, 'String: Your dataset directory')

# The number of images in the validation set. You would have to know the total number of examples in advance. This is essentially your evaluation dataset.
flags.DEFINE_float('validation_size', 0.25, 'Float: The proportion of examples in the dataset to be used for validation')

# The number of shards to split the dataset into
flags.DEFINE_integer('num_shards', 1, 'Int: Number of shards to split the TFRecord files')

# Seed for repeatability.
flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

#Output filename for the naming the TFRecord file
flags.DEFINE_string('tfrecord_filename', None, 'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS

def main():

    #==============================================================CHECKS==========================================================================
    #Check if there is a tfrecord_filename entered
    if not FLAGS.tfrecord_filename:
        raise ValueError('tfrecord_filename is empty. Please state a tfrecord_filename argument.')

    #Check if there is a dataset directory entered
    if not FLAGS.dataset_dir:
        raise ValueError('dataset_dir is empty. Please state a dataset_dir argument.')

    #If the TFRecord files already exist in the directory, then exit without creating the files again
    if _dataset_exists(dataset_dir = FLAGS.dataset_dir, _NUM_SHARDS = FLAGS.num_shards, output_filename = FLAGS.tfrecord_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return None
    #==============================================================END OF CHECKS===================================================================

    #Get two list of photo_filenames like ['123.jpg', '456.jpg'...] of color images and depth images and a list of sorted class names from parsing the subdirectories.
    photo_filenames_color, photo_filenames_depth, class_names = _get_filenames_and_classes(FLAGS.dataset_dir)
    photo_filenames = map(list,zip(photo_filenames_color, photo_filenames_depth))

    #Refer each of the class name to a specific integer number for predictions later
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    #Find the number of validation examples we need
    num_validation = int(FLAGS.validation_size * len(photo_filenames))

    # Divide the training datasets into train and test:
    random.seed(FLAGS.random_seed)
    random.shuffle(photo_filenames)
    photo_filenames_color, photo_filenames_depth = zip(*photo_filenames)
    training_filenames_color = photo_filenames_color[num_validation:]
    validation_filenames_color = photo_filenames_color[:num_validation]
    training_filenames_depth = photo_filenames_depth[num_validation:]
    validation_filenames_depth = photo_filenames_depth[:num_validation]

    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames_color, training_filenames_depth, class_names_to_ids, dataset_dir = FLAGS.dataset_dir, tfrecord_filename = FLAGS.tfrecord_filename, _NUM_SHARDS = FLAGS.num_shards)
    _convert_dataset('validation', validation_filenames_color, validation_filenames_depth, class_names_to_ids, dataset_dir = FLAGS.dataset_dir, tfrecord_filename = FLAGS.tfrecord_filename, _NUM_SHARDS = FLAGS.num_shards)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    write_label_file(labels_to_class_names, FLAGS.dataset_dir)
    write_file_pattern_file(FLAGS.tfrecord_filename + '_%s_*.tfrecord', FLAGS.dataset_dir)

    print('\nFinished converting the %s dataset!' % (FLAGS.tfrecord_filename))

if __name__ == "__main__":
    main()
