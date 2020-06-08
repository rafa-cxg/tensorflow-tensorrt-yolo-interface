#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
## Get calibration data

# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import random

# For our custom calibrator
from calibrator import load_mnist_data, load_mnist_labels, MNISTEntropyCalibrator

# For ../common.py
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
import common

TRT_LOGGER = trt.Logger()


class ModelData(object):
    DEPLOY_PATH = "deploy.prototxt"
    MODEL_PATH = "mnist_lenet.caffemodel"
    OUTPUT_NAME = "prob"
    # The original model is a float32 one.
    DTYPE = trt.float32


# This function builds an engine from a Caffe model.
def build_int8_engine(deploy_file, model_file, calib, batch_size=32):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.CaffeParser() as parser:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size.
        builder.max_batch_size = batch_size
        builder.max_workspace_size = common.GiB(1)
        builder.int8_mode = True
        builder.int8_calibrator = calib
        # Parse Caffe model
        model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=ModelData.DTYPE)
        network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME))
        # Build engine and do int8 calibration.
        return builder.build_cuda_engine(network)


def check_accuracy(context, batch_size, test_set, test_labels):
    inputs, outputs, bindings, stream = common.allocate_buffers(context.engine)

    num_correct = 0
    num_total = 0

    batch_num = 0
    for start_idx in range(0, test_set.shape[0], batch_size):
        batch_num += 1
        if batch_num % 10 == 0:
            print("Validating batch {:}".format(batch_num))
        # If the number of images in the test set is not divisible by the batch size, the last batch will be smaller.
        # This logic is used for handling that case.
        end_idx = min(start_idx + batch_size, test_set.shape[0])
        effective_batch_size = end_idx - start_idx

        # Do inference for every batch.
        inputs[0].host = test_set[start_idx:start_idx + effective_batch_size]
        [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=effective_batch_size)

        # Use argmax to get predictions and then check accuracy
        preds = np.argmax(output.reshape(32, 10)[0:effective_batch_size], axis=1)
        labels = test_labels[start_idx:start_idx + effective_batch_size]
        num_total += effective_batch_size
        num_correct += np.count_nonzero(np.equal(preds, labels))

    percent_correct = 100 * num_correct / float(num_total)
    print("Total Accuracy: {:}%".format(percent_correct))
from official.resnet import imagenet_preprocessing  # pylint: disable=g-bad-import-order
import imghdr
import tensorflow as tf
def preprocess_image(file_name, output_height=224, output_width=224,
                     num_channels=3):
  """Run standard ImageNet preprocessing on the passed image file.
  Args:
    file_name: string, path to file containing a JPEG image
    output_height: int, final height of image
    output_width: int, final width of image
    num_channels: int, depth of input image
  Returns:
    Float array representing processed image with shape
      [output_height, output_width, num_channels]
  Raises:
    ValueError: if image is not a JPEG.
  """
  if imghdr.what(file_name) != "jpeg":
    raise ValueError("At this time, only JPEG images are supported. "
                     "Please try another image.")

  image_buffer = tf.read_file(file_name)
  normalized = imagenet_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      bbox=None,
      output_height=output_height,
      output_width=output_width,
      num_channels=num_channels,
      is_training=False)

  with tf.Session(config=get_gpu_config()) as sess:
    result = sess.run([normalized])

  return result[0]


def batch_from_image(file_name, batch_size, output_height=224, output_width=224,
                     num_channels=3):
  """Produce a batch of data from the passed image file.
  Args:
    file_name: string, path to file containing a JPEG image
    batch_size: int, the size of the desired batch of data
    output_height: int, final height of data
    output_width: int, final width of data
    num_channels: int, depth of input data
  Returns:
    Float array representing copies of the image with shape
      [batch_size, output_height, output_width, num_channels]
  """
  image_array = preprocess_image(
      file_name, output_height, output_width, num_channels)

  tiled_array = np.tile(image_array, [batch_size, 1, 1, 1])
  return tiled_array



def main():
    # _, data_files = common.find_sample_data(description="Runs a Caffe MNIST network in Int8 mode", subfolder="mnist", find_files=["t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", "train-images-idx3-ubyte", ModelData.DEPLOY_PATH, ModelData.MODEL_PATH])
    # [test_set, test_labels, train_set, deploy_file, model_file] = data_files
    #
    # # Now we create a calibrator and give it the location of our calibration data.
    # # We also allow it to cache calibration data for faster engine building.
    # calibration_cache = "mnist_calibration.cache"
    # calib = MNISTEntropyCalibrator(test_set, cache_file=calibration_cache)
    #
    # # Inference batch size can be different from calibration batch size.
    # batch_size = 32
    # with build_int8_engine(deploy_file, model_file, calib, batch_size) as engine, engine.create_execution_context() as context:
    #     # Batch size for inference can be different than batch size used for calibration.
    #     check_accuracy(context, batch_size, test_set=load_mnist_data(test_set), test_labels=load_mnist_labels(test_labels))
    import requests
    from PIL import Image
    # IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'
    # data = requests.get(IMAGE_URL, stream=True).content

    # data = Image.open('cat.jpg')
    data = batch_from_image('cat.jpg', 1)
    # Convert and calibrate model
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    import numpy as np
    import tensorflow as tf
    # Load and convert a frozen graph
    graph_def = tf.GraphDef()
    with tf.gfile.GFile("fire4.pb", 'rb') as f:
        graph_def.ParseFromString(f.read())
    # dataset = tf.data.TFRecordDataset('fire-data')
    # iterator = dataset.make_one_shot_iterator()
    # features = iterator.get_next()
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    converter = trt.TrtGraphConverter(input_graph_def=graph_def, nodes_blacklist=['conv_81/BiasAdd', 'conv_93/BiasAdd', 'conv_105/BiasAdd'], precision_mode='INT8')
    graph_def = converter.convert()

    def input_map_fn():
        return {'input_1:0': features}
    calibrated_graph_def = converter.calibrate(
        fetch_names=['conv_81/BiasAdd', 'conv_93/BiasAdd', 'conv_105/BiasAdd'],
        num_runs=1,
        feed_dict_fn=lambda: {'input_1:0': np.array([data])}
    )
    converter.save('fire48.pb')
if __name__ == '__main__':
    main()
