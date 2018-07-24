import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import vgg_preprocessing
from vgg_test import vgg_16, vgg_19, vgg_arg_scope
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import cv2
slim = tf.contrib.slim
flags = tf.app.flags

# export LANGUAGE=en_US.UTF-8
# export LC_ALL=en_US.UTF-8

#================ DATASET INFORMATION ======================
#State dataset directory where the tfrecord files are located
flags.DEFINE_string('dataset_dir', '/home/lou/Dataset/', 'String: dataset directory')

#State where your log file is at. If it doesn't exist, create it.
flags.DEFINE_string('log_dir', './vgg_Logs', 'String: directory in which the logs should be written')

#State where your checkpoint file is
flags.DEFINE_string('checkpoint_file', './checkpoints/vgg_16.ckpt', 'String: path of the checkpoint')

#State the image size you're resizing your images to. We will use the default inception size of 299.
image_size = 224

#State the number of classes to predict:
num_classes = 2

FLAGS = flags.FLAGS
#State the labels file and read it
# labels_file = '/home/lou/Feature_Embedding/2_Workspace/py3_virtualenv/training/Dataset/labels.txt'
# labels = open(labels_file, 'r')
flags.DEFINE_string('labels_file', '/home/lou/Dataset/labels.txt', 'String: path of the labels file')
labels = open(FLAGS.labels_file, 'r')

#Create a dictionary to refer each label to their string name
labels_to_name = {}
for line in labels:
    label, string_name = line.split(':')
    string_name = string_name[:-1] #Remove newline
    labels_to_name[int(label)] = string_name

#Create the file pattern of your TFRecord files so that it could be recognized later on
file_pattern = 'Feature_Embedding_%s_*.tfrecord'

#Create a dictionary that will help people understand your dataset better. This is required by the Dataset class later.
items_to_descriptions = {
    'image_color': 'A 3-channel RGB coloured grasp image that is either grasp or not_grasp.',
    'image_depth': 'A 1-channel Depth image that is either grasp or not_grasp.',
    'label': 'A label that is as such -- 0:grasp, 1:not_grasp'
}


#================= TRAINING INFORMATION ==================
#State the number of epochs to train
num_epochs = 30

#State your batch size
batch_size = 20

#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.0002
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 2

#============== DATASET LOADING ======================
# We now create a function that creates a Dataset class which will give us many TFRecord files to feed in the examples into a queue in parallel.
def get_split(split_name, dataset_dir, file_pattern=file_pattern):
    '''
    Obtains the split - training or validation - to create a Dataset class for feeding the examples into a queue later on. This function will
    set up the decoder and dataset information all into one Dataset class so that you can avoid the brute work later on.
    Your file_pattern is very important in locating the files later. 

    INPUTS:
    - split_name(str): 'train' or 'validation'. Used to get the correct data split of tfrecord files
    - dataset_dir(str): the dataset directory where the tfrecord files are located
    - file_pattern(str): the file name structure of the tfrecord files in order to get the correct data

    OUTPUTS:
    - dataset (Dataset): A Dataset class object where we can read its various components for easier batch creation later.
    '''

    #First check whether the split_name is train or validation
    if split_name not in ['train', 'validation']:
        raise ValueError('The split_name %s is not recognized. Please input either train or validation as the split_name' % (split_name))

    #Create the full path for a general file_pattern to locate the tfrecord_files
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    #Count the total number of examples in all of these shard
    num_samples = 0
    file_pattern_for_counting = 'Feature_Embedding_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    #Create a reader, which must be a TFRecord reader in this case
    reader = tf.TFRecordReader

    #Create the keys_to_features dictionary for the decoder
    keys_to_features = {
      'image/color': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/depth': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }

    #Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
    'color': slim.tfexample_decoder.Image('image/color'),
    'depth': slim.tfexample_decoder.Image('image/depth'),
    'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    #Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    #Create the labels_to_name file
    labels_to_name_dict = labels_to_name

    #Actually create the dataset
    dataset = slim.dataset.Dataset(
        data_sources = file_pattern_path,
        decoder = decoder,
        reader = reader,
        num_readers = 4,
        num_samples = num_samples,
        num_classes = num_classes,
        labels_to_name = labels_to_name_dict,
        items_to_descriptions = items_to_descriptions)

    return dataset


def load_batch(dataset, batch_size, height=image_size, width=image_size, is_training=True):
    '''
    Loads a batch for training.

    INPUTS:
    - dataset(Dataset): a Dataset class object that is created from the get_split function
    - batch_size(int): determines how big of a batch to train
    - height(int): the height of the image to resize to during preprocessing
    - width(int): the width of the image to resize to during preprocessing
    - is_training(bool): to determine whether to perform a training or evaluation preprocessing

    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).

    '''
    # First create the data_provider object
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity = 24 + 3 * batch_size,
        common_queue_min = 24)

    # Obtain the raw image using the get method
    raw_image_color, raw_image_depth, label = data_provider.get(['color', 'depth', 'label'])

    # Perform the correct preprocessing for this image depending if it is training or evaluating
    image_color = vgg_preprocessing.preprocess_image(raw_image_color, height, width, is_training)
    image_depth = tf.image.resize_images(raw_image_depth,(56,56))
    image_depth = vgg_preprocessing.preprocess_image(image_depth, 56, 56, is_training)

    # As for the raw images, we just do a simple reshape to batch it up
    raw_image_color = tf.expand_dims(raw_image_color, 0)
    raw_image_color = tf.image.resize_nearest_neighbor(raw_image_color, [height, width])
    raw_image_color = tf.squeeze(raw_image_color)

    raw_image_depth = tf.expand_dims(raw_image_depth, 0)
    raw_image_depth = tf.image.resize_nearest_neighbor(raw_image_depth, [height, width])
    raw_image_depth = tf.squeeze(raw_image_depth)

    # Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    image_color, raw_image_color, image_depth, raw_image_depth, labels = tf.train.batch(
        [image_color, raw_image_color, image_depth, raw_image_depth, label],
        batch_size = batch_size,
        num_threads = 4,
        capacity = 4 * batch_size,
        allow_smaller_final_batch = True)

    return image_color, raw_image_color, image_depth, raw_image_depth, labels

def run():
    # Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)

    # ======================= TRAINING PROCESS =========================
    # Now we start to construct the graph and build our model
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO) #Set the verbosity to INFO level

        # First create the dataset and load one batch
        dataset = get_split('train', FLAGS.dataset_dir, file_pattern)
        images_color, _, images_depth, _, labels = load_batch(dataset, batch_size=batch_size)

        # Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = int(dataset.num_samples / batch_size) # because the num_samples of dataset_color equals that of dataset_depth
        num_steps_per_epoch = num_batches_per_epoch # Because one step is one batch processed
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

        # Create the model inference
        with slim.arg_scope(vgg_arg_scope()):
            logits, end_points = vgg_16(images_color, images_depth, num_classes = dataset.num_classes, is_training = True)

        # Define the scopes that you want to exclude for restoration
        # not sure what to exclude for VGG
        exclude = ['vgg_16/fc8', 'vgg_16/conv3']
        # variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
        variables_to_restore = slim.get_variables_to_restore(exclude = exclude)

        # Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)

        # Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
        loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits)
        # slim.losses.softmax_cross_entropy(predictions, labels)
        total_loss = tf.losses.get_total_loss()    # obtain the regularization losses as well

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        # Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate = initial_learning_rate,
            global_step = global_step,
            decay_steps = decay_steps,
            decay_rate = learning_rate_decay_factor,
            staircase = True)

        # Now we can define the optimizer that takes on the learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)

        # Create the train_op.
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(end_points['vgg_16/fc8'], 1)
        probabilities = end_points['vgg_16/fc8']
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        metrics_op = tf.group(accuracy_update, probabilities)

        # predictions = vgg_16(images_color, images_depth, is_training=False)
        # accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)

        # Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        tf.summary.histogram('probabilities', probabilities)
        tf.summary.histogram('Logits', logits)
        tf.summary.histogram('Predictions', predictions)
        my_summary_op = tf.summary.merge_all()

        # Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            # Check the time for each sess run
            start_time = time.time()
            total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
            time_elapsed = time.time() - start_time

            # Run the logging to print some results
            logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

            return total_loss, global_step_count

        # Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, FLAGS.checkpoint_file)
        intermediate_saver = tf.train.Saver(max_to_keep=3)

        # Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir = FLAGS.log_dir, summary_op = None, init_fn = restore_fn)


        # Run the managed session
        with sv.managed_session() as sess:
            for step in range(num_steps_per_epoch * num_epochs):
            # for step in xrange(1):
                # At the start of every epoch, show the vital information:
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, num_epochs)
                    learning_rate_value, accuracy_value = sess.run([lr, accuracy])
                    logging.info('Current Learning Rate: %s', learning_rate_value)
                    logging.info('Current Streaming Accuracy: %s', accuracy_value)

                    # optionally, print your logits and predictions for a sanity check that things are going fine.
                    logits_value, probabilities_value, predictions_value, labels_value = sess.run([logits, probabilities, predictions, labels])
                    print('logits: \n', logits_value)
                    print('Probabilities: \n', probabilities_value)
                    print('predictions: \n', predictions_value)
                    print('Labels:\n:', labels_value)

                # Log the summaries every 10 step.
                if step % 10 == 0:
                    loss, _ = train_step(sess, train_op, sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)

                # If not, simply run the training step
                else:
                    loss, _ = train_step(sess, train_op, sv.global_step)

		# Save checkpoint for ever fifth epoch
                if step%int(num_steps_per_epoch) == 0:
                    intermediate_saver.save(sess, './vgg_intermediate_Logs/model.ckpt', global_step = step)


            # We log the final training loss and accuracy
            logging.info('Final Loss: %s', loss)
            logging.info('Final Accuracy: %s', sess.run(accuracy))

            # Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            # saver.save(sess, "./flowers_model.ckpt")
            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)


if __name__ == '__main__':
    run()