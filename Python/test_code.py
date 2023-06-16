import tensorflow as tf

dWI1 = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])
dWI41 = tf.constant([[16.0, 17.0, 18.0], [19.0, 20.0, 21.0], [22.0, 23.0, 24.0], [25.0, 26.0, 27.0], [28.0, 29.0, 30.0]])
dWI42 = tf.constant([[31.0, 32.0, 33.0], [34.0, 35.0, 36.0], [37.0, 38.0, 39.0], [40.0, 41.0, 42.0], [43.0, 44.0, 45.0]])

# Create 3D tensor
Hess = tf.transpose(tf.stack([dWI1, dWI41, dWI42]), perm=[1, 0, 2])
Hess_t= tf.transpose(Hess, perm=[0, 2, 1])

# print(Hess-Hess_t)
print(tf.reduce_mean(tf.reduce_sum(Hess - Hess_t, axis=[-2,-1])))

Delta_k = []
for i in range(Hess.shape[0]):
    # Get the 3x3 matrix at the current row
    matrix = Hess[i]

    # Calculate the values
    first_column = matrix[0, 0]
    second_column = tf.linalg.det(matrix[:2, :2])
    third_column = tf.linalg.det(matrix)

    # Append the values as a row to the new array
    Delta_k.append([first_column, second_column, third_column])

# Convert the list to a TensorFlow tensor
Delta_k = tf.convert_to_tensor(Delta_k)
L_positive = tf.reduce_mean(tf.reduce_sum(tf.maximum(-Delta_k, 0), axis=1))

print(Delta_k)
print(L_positive)