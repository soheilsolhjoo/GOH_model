import tensorflow as tf
import numpy as np

# @tf.function
def calculate_determinant(matrix):
    matrix_size = tf.shape(matrix)[0]
    tf.print("Matrix size:", matrix_size)

    if matrix_size == 2:
        a = matrix[0, 0]
        b = matrix[0, 1]
        c = matrix[1, 0]
        d = matrix[1, 1]

        determinant = a * d - b * c

    elif matrix_size >= 3:
        determinant = tf.constant(0.0, dtype=tf.float32)
        for j in range(matrix_size):
            submatrix = tf.concat([matrix[1:, :j], matrix[1:, j+1:]], axis=1)
            determinant += tf.cast(((-1) ** j), dtype=tf.float32) * matrix[0, j] * calculate_determinant(submatrix)

    else:
        raise ValueError("Unsupported matrix size. Expected at least 2x2.")

    return determinant


dWI1 = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])
dWI41 = tf.constant([[16.0, 17.0, 18.0], [19.0, 20.0, 21.0], [22.0, 23.0, 24.0], [25.0, 26.0, 27.0], [28.0, 29.0, 30.0]])
dWI42 = tf.constant([[31.0, 32.0, 33.0], [34.0, 35.0, 36.0], [37.0, 38.0, 39.0], [40.0, 41.0, 42.0], [43.0, 44.0, 45.0]])

# Create 3D tensor
Hess = tf.transpose(tf.stack([dWI1, dWI41, dWI42]), perm=[1, 0, 2])

# Distribute the matrix along the first dimension
matrix_list = tf.unstack(Hess)

Delta_k = []
for i in range(1, tf.shape(Hess)[0]):
    delta_i = [
        matrix_list[i][0, 0],
        calculate_determinant(matrix_list[i][:2, :2]),
        calculate_determinant(matrix_list[i])
    ]
    Delta_k.append(delta_i)

Delta_k = tf.stack(Delta_k)

print(Delta_k)
print(tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.math.negative(Delta_k), 0), axis=1)))
