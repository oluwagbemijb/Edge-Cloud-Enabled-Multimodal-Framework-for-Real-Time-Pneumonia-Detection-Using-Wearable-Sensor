"""
domain_adaptation.py
Implementation of the Gradient Reversal Layer (GRL) for Adversarial Domain Adaptation.
As described in: Multimodal Deep Learning for Pneumonia Detection Using Wearable Sensors
"""

import tensorflow as tf
from tensorflow.keras import layers

class GradientReversalLayer(layers.Layer):
    """
    Custom Layer for Adversarial Domain Adaptation.
    
    Forward Pass: Acts as an identity function (passes data through unchanged).
    Backward Pass: Multiplies the gradient by a negative scalar (-lambda).
    
    This forces the feature extractor to produce features that the domain 
    classifier cannot distinguish, promoting domain-invariant representations 
    across heterogeneous datasets (MIMIC, Coswara, Local Wearable).
    """
    def __init__(self, hp_lambda=1.0, name="grl_layer", **kwargs):
        super(GradientReversalLayer, self).__init__(name=name, **kwargs)
        self.hp_lambda = hp_lambda

    @tf.custom_gradient
    def reverse_gradient(self, x):
        """
        Defines the custom gradient behavior.
        """
        # Forward pass: Return input as is
        def grad(dy):
            # Backward pass: Reverse the gradient and scale by lambda
            return -dy * self.hp_lambda
        
        return x, grad

    def call(self, x, training=None):
        """
        Executes the layer during the forward pass.
        """
        return self.reverse_gradient(x)

    def get_config(self):
        """
        Ensures the layer can be saved and reloaded with Keras models.
        """
        config = super(GradientReversalLayer, self).get_config()
        config.update({"hp_lambda": self.hp_lambda})
        return config


def compute_domain_loss(y_true, y_pred):
    """
    Standard Binary Cross-Entropy used for the Domain Discriminator.
    Referenced in Section 4.1 of the journal.
    """
    bce = tf.keras.losses.CategoricalCrossentropy()
    return bce(y_true, y_pred)


# Usage Example within the Multimodal Framework (Conceptual)
if __name__ == "__main__":
    # Simulate a fused feature vector from Section 3.4
    # (128 CNN + 128 Bi-LSTM + 32 Static = 288 units)
    mock_fused_features = tf.random.normal([32, 288])
    
    # 1. Initialize the GRL
    grl = GradientReversalLayer(hp_lambda=1.0)
    
    # 2. Apply GRL to features before the Domain Classifier
    domain_features = grl(mock_fused_features)
    
    # 3. Domain Classifier (Distinguishes between MIMIC, Coswara, and Local)
    # Output: 3 classes (one-hot encoded)
    domain_output = layers.Dense(3, activation='softmax', name="domain_classifier")(domain_features)
    
    print(f"Input Feature Shape: {mock_fused_features.shape}")
    print(f"Domain Classifier Output: {domain_output.shape}")
    print("GRL successfully integrated for Adversarial Training.")