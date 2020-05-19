"""
Core Module for Grad CAM Algorithm
"""
import cv2
import numpy as np
import tensorflow as tf

from tf_explain.utils.display import grid_display, heatmap_display, image_to_uint_255
from tf_explain.utils.saver import save_rgb


class GradCAM:

    """
    Perform Grad CAM algorithm for a given input

    Paper: [Grad-CAM: Visual Explanations from Deep Networks
            via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
    """

    def explain(
        self,
        validation_data,
        model,
        first,
        batch_mode,
        class_index,
        layer_name=None,
        colormap=cv2.COLORMAP_VIRIDIS,
        image_weight=0.8,
    ):
        """
        Compute GradCAM for a specific class index.

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            layer_name (str): Targeted layer for GradCAM. If no layer is provided, it is
                automatically infered from the model architecture.
            colormap (int): OpenCV Colormap to use for heatmap visualization
            image_weight (float): An optional `float` value in range [0,1] indicating the weight of
                the input image to be overlaying the calculated attribution maps. Defaults to `0.7`.

        Returns:
            numpy.ndarray: Grid of all the GradCAM
        """
        if batch_mode:
           
           if first:
                    images, class_index = validation_data.__getitem__(0)
           else:
                    images, class_index = validation_data.__getitem__(validation_data.batch_index)
           
           class_index = class_index.flatten()
           class_index = np.where(class_index == 1)
           class_index = np.asarray([num - (i*4) for i,num in enumerate(class_index[0])])
        else:
           images, _ = validation_data.__getitem__(0)
        if layer_name is None:
            layer_name = self.infer_grad_cam_target_layer(model)

        outputs, guided_grads, class_index = GradCAM.get_gradients_and_filters(
            model, images, layer_name, class_index
        )

        cams = GradCAM.generate_ponderated_output(outputs, guided_grads, class_index)
        heatmaps = np.array(
            [
                # not showing the actual image if image_weight=0
                heatmap_display(cam.numpy(), image, colormap, image_weight)
                for cam, image in zip(cams, images)
            ]
        )
        
        for i in range(len(images)):
            
            heatmaps = np.concatenate([heatmaps, [image_to_uint_255(images[i])]])


            #heatmapsnp.array(cv2.cvtColor(image_to_uint_255(images[i]), cv2.COLOR_BGR2RGB))
            #heatmaps.append(cv2.cvtColor(image_to_uint_255(images[i]), cv2.COLOR_BGR2RGB))

        grid = grid_display(heatmaps,2,len(images))

        return grid

    @staticmethod
    def infer_grad_cam_target_layer(model):
        """
        Search for the last convolutional layer to perform Grad CAM, as stated
        in the original paper.

        Args:
            model (tf.keras.Model): tf.keras model to inspect

        Returns:
            str: Name of the target layer
        """
        for layer in reversed(model.layers):
            # Select closest 4D layer to the end of the network.
            if len(layer.output_shape) == 4:
                return layer.name

        raise ValueError(
            "Model does not seem to contain 4D layer. Grad CAM cannot be applied."
        )

    @staticmethod
    @tf.function
    def get_gradients_and_filters(model, images, layer_name, class_index):
        """
        Generate guided gradients and convolutional outputs with an inference.

        Args:
            model (tf.keras.Model): tf.keras model to inspect
            images (numpy.ndarray): 4D-Tensor with shape (batch_size, H, W, 3)
            layer_name (str): Targeted layer for GradCAM
            class_index (int): Index of targeted class

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: (Target layer outputs, Guided gradients)
        """
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )
        
        
        
        
        with tf.GradientTape(persistent=True) as tape:
                    inputs = tf.cast(images, tf.float32)
                    conv_outputs, predictions = grad_model(inputs)
                    
                    grads = []
                    guided_grads = []
                    for i in range(len(class_index)):
                        loss = predictions[:, class_index[i]]
                        grads.append(tape.gradient(loss, conv_outputs))
                    #grads = tf.convert_to_tensor(grads, dtype=tf.float32)
                        guided_grads.append(
                             tf.cast(conv_outputs > 0, "float32") * tf.cast(grads[i] > 0, "float32") * grads[i]
                            )
        return conv_outputs, guided_grads, class_index

    @staticmethod
    def generate_ponderated_output(outputs, grads, class_index):
        """
        Apply Grad CAM algorithm scheme.

        Inputs are the convolutional outputs (shape WxHxN) and gradients (shape WxHxN).
        From there:
            - we compute the spatial average of the gradients
            - we build a ponderated sum of the convolutional outputs based on those averaged weights

        Args:
            output (tf.Tensor): Target layer outputs, with shape (batch_size, Hl, Wl, Nf),
                where Hl and Wl are the target layer output height and width, and Nf the
                number of filters.
            grads (tf.Tensor): Guided gradients with shape (batch_size, Hl, Wl, Nf)

        Returns:
            List[tf.Tensor]: List of ponderated output of shape (batch_size, Hl, Wl, 1)
        """

        maps = []
       
        for i in  range(len(grads)):
            for output, grad, j in zip(outputs, grads[i], range(tf.shape(grads[i])[0])):
                 if j == class_index[i]:
                    maps.append(GradCAM.ponderate_output(output, grad))
                    break
                   
        

        return maps

    @staticmethod
    def ponderate_output(output, grad):
        """
        Perform the ponderation of filters output with respect to average of gradients values.

        Args:
            output (tf.Tensor): Target layer outputs, with shape (Hl, Wl, Nf),
                where Hl and Wl are the target layer output height and width, and Nf the
                number of filters.
            grads (tf.Tensor): Guided gradients with shape (Hl, Wl, Nf)

        Returns:
            tf.Tensor: Ponderated output of shape (Hl, Wl, 1)
        """

        weights = tf.reduce_mean(grad, axis=(0, 1))
        # Perform ponderated sum : w_i * output[:, :, i]
        cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
        
        return cam

    def save(self, grid, output_dir, output_name):
        """
        Save the output to a specific dir.

        Args:
            grid (numpy.ndarray): Grid of all the heatmaps
            output_dir (str): Output directory path
            output_name (str): Output name
        """
        save_rgb(grid, output_dir, output_name)
