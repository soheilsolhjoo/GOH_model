import tensorflow as tf

# Scalar tensor
scalar_tensor = tf.constant(5)  # Shape: ()

# 1D tensor
tensor_1d = tf.constant([2])  # Shape: (1,)
print(tf.squeeze(tensor_1d))

# Addition
result = scalar_tensor + tensor_1d

# Print the result
print("Result:", result)
