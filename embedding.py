import tensorflow as tf


class Embedding():
	def __init__(self, trainable=False):
		self.trainable = trainable

	def init(self):
		self.embeddings = tf.Variable(initial_value = tf.zeros([self.vocab_size, self.vector_size]), trainable = self.trainable, name = "embeddings")
		self.embeddings_placeholder = tf.placeholder(dtype = tf.float32, shape = (self.vocab_size, self.vector_size), name = "embeddings_placeholder")
		self.embeddings_op = self.embeddings.assign(self.embeddings_placeholder)

	def load(self, filename):
		self.all_vectors = []
		self.mappings = {}
		counter = 0
		self.mappings['PAD'] = 0
		self.mappings['EOS'] = 1
		self.mappings['UNK'] = 2
		self.all_vectors.append(None)
		self.all_vectors.append(None)
		self.all_vectors.append(None)
		counter = len(self.all_vectors)
		start = counter

		with open(filename, 'r') as f:
			for line in f.readlines():
				try:
					vector = line.split(' ')
					self.mappings[vector[0]] = counter
					vector = [float(x) for i, x in enumerate(vector) if i != 0]
					self.all_vectors.append(vector)
					counter += 1
				except Exception as e:
					print(e)
					print("Error parsing line", line)

		self.vocab_size = len(self.all_vectors)
		self.vector_size = len(self.all_vectors[start])
		self.all_vectors[0] = [-1.0 for i in range(self.vector_size)]
		self.all_vectors[1] = [1.0 for i in range(self.vector_size)]
		self.all_vectors[2] = [0.0 for i in range(self.vector_size)]

	def lookup(self, ids):
			return tf.nn.embedding_lookup(self.embeddings, ids)

	def get_token(self, token):
		return self.mappings.get(token, 0) if token != '.' else 1


if __name__ == "__main__":
	emb = Embedding()
	emb.load("../final_vectors.txt")
	emb.init()
