import tensorflow as tf
import numpy as np

#operações
tensor= tf.constant([[1,2], [3,4]])
tensor_2= tf.constant([[2,3], [7,8]])
print(tensor)
print(tensor+2)
print(tensor*10)
print(np.square(tensor)) #potencia 2
print(np.sqrt(tensor)) #raiz quadrada
print(np.dot(tensor, tensor_2)) #produto escalar

#strings
stringtf= tf.constant("TensorFlow")
print(stringtf)
print(tf.strings.length(stringtf))
tf.strings.unicode_decode(stringtf, "UTF8")
string_array=tf.constant(["TensorFlow", "Deep Learning", "AI"])

# Iterating through the TF string array
for string in string_array:
  print(string)


