import os
import sys
import time
import argparse
import numpy as np
import tensorrt as trt
from cuda import cudart
import torch
import torchvision.transforms as transforms
from PIL import Image

import mani_skill.trt_utils.common as common

#from image_batcher import ImageBatcher
#from visualize import visualize_detections


class TensorRTInfer:
    """
    Implements inference for the EfficientDet TensorRT engine.
    """

    def __init__(self, engine_path, batch_size):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context
        self.batch_size = batch_size
        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            shape = self.context.get_tensor_shape(name)
            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_tensor_profile_shape(name, 0)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                # self.context.set_input_shape(name, profile_shape[1])
                self.context.set_input_shape(name, trt.Dims([batch_size, 2, 384,768]))
                shape = self.context.get_tensor_shape(name)
            if is_input:
                self.batch_size = shape[0]
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            print(
                "{} '{}' with shape {} and dtype {}".format(
                    "Input" if is_input else "Output",
                    binding["name"],
                    binding["shape"],
                    binding["dtype"],
                )
            )

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o["shape"], o["dtype"]))
        return specs
    
    def preprocess(self, image_left, image_right):
        normal_mean_var = {'mean': [0.485],
                            'std': [0.229]}

        infer_transform = transforms.Compose([transforms.Normalize(**normal_mean_var)])

        # image_left = image_left.resize([768,384])
        # image_right = image_right.resize([768,384])
        #image_np = np.array(image)

        image_left = image_left.type(torch.float32)/255.
        image_right = image_right.type(torch.float32)/255.
        image_left = image_left.permute(0,3,1,2)
        image_right = image_right.permute(0,3,1,2)
        image_left = transforms.functional.rgb_to_grayscale(image_left)
        image_right = transforms.functional.rgb_to_grayscale(image_right)
        image_left = infer_transform(image_left)
        image_right = infer_transform(image_right)
        image = torch.cat([image_left, image_right], dim=1)


        
        image_batch = image.cpu().numpy()
        image_batch = image_batch.ravel()
        return image_batch

    def infer(self, batch):
        """
        Execute inference on a batch of images.
        :param batch: A numpy array holding the image batch.
        :return A list of outputs as numpy arrays.
        """
        # Copy I/O and Execute
        common.memcpy_host_to_device(self.inputs[0]["allocation"], batch)
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            common.memcpy_device_to_host(
                self.outputs[o]["host_allocation"], self.outputs[o]["allocation"]
            )
        return [o["host_allocation"] for o in self.outputs]

    def process(self, image_left, image_right):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """
        # Run inference
        batch = self.preprocess(image_left, image_right)
        outputs = self.infer(batch)
        outputs = np.reshape(outputs, (self.batch_size, 384, 768))
        # Process the results
        
        return outputs


def main():
    engine_file = "/home/jianyu/jianyu/pythonproject/fadnet_jetson/test_ir.trt"
    input_file_left  = "/home/jianyu/jianyu/pythonproject/fadnet/sample_data/left.png"
    input_file_right = "/home/jianyu/jianyu/pythonproject/fadnet/sample_data/right.png"
    output_file = "trt_output_pc_ir_test"


    trt_infer = TensorRTInfer(engine_file)
    test_image_left = Image.open(input_file_left).convert('L')
    test_image_right = Image.open(input_file_right).convert('L')
    batch = trt_infer.preprocess(test_image_left, test_image_right)
    start_time = time.time()
    detections = trt_infer.process(batch)
    time_elapsed = time.time() - start_time
    
    pred = np.reshape(detections, (16, 384, 768))
    for i in range(pred.shape[0]):
        cimg = pred[i,...]
        img = (cimg*256).astype('uint16')
        img = Image.fromarray(img)
        img.save(output_file+f"_{i}.png")
    #common.free_buffers(inputs, outputs, stream)


    print()
    print(f"Finished Processing in {time_elapsed}s")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-e",
    #     "--engine",
    #     default=None,
    #     required=True,
    #     help="The serialized TensorRT engine",
    # )
    # parser.add_argument(
    #     "-i", "--input", default=None, help="Path to the image or directory to process"
    # )
    # parser.add_argument(
    #     "-o",
    #     "--output",
    #     default=None,
    #     help="Directory where to save the visualization results",
    # )
    # parser.add_argument(
    #     "-l",
    #     "--labels",
    #     default="./labels_coco.txt",
    #     help="File to use for reading the class labels from, default: ./labels_coco.txt",
    # )
    # parser.add_argument(
    #     "-t",
    #     "--nms_threshold",
    #     type=float,
    #     help="Override the score threshold for the NMS operation, if higher than the built-in threshold",
    # )
    # args = parser.parse_args()
    main()