import sys
import numpy as np
import random
import embedding as emb
import tensorflow as tf
LOGDIR="./tensorflow_logs"

class Seq2Seq(object):
	def __init__(self, embedding, size, encoder_stack_size = 1, decoder_stack_size = 1, attention_depth = 4):
		self.encoder_hidden_size = size
		self.decoder_hidden_size = size
		self.attention_depth = attention_depth
		self.encoder_stack_size = encoder_stack_size
		self.decoder_stack_size = decoder_stack_size

		self.embedding = embedding
		self._make_graph()

	def _make_graph(self):
		self.encoder_cell = self._make_cell(self.encoder_hidden_size, self.encoder_stack_size)
		self.decoder_cell = self._make_cell(self.decoder_hidden_size, self.decoder_stack_size)
		self._init_placeholders()
		self._init_decoder_inputs()
		self._init_encoder_inputs()
		self._init_encoder()
		self._init_decoder()
		self._init_decoder_train()
		self._init_decoder_inference()
		self._init_loss()
		self._init_accuracy()

	def _make_cell(self, hidden_units, stack_size = 1, dropout = 0.5):
		def get_rnn_cell(hidden_size):
			lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)
			dropout_cell = tf.contrib.rnn.DropoutWrapper(cell = lstm_cell, input_keep_prob = 0.5)
			return dropout_cell

		multi_rnn_cell = tf.contrib.rnn.MultiRNNCell([get_rnn_cell(hidden_units) for i in range(stack_size)])
		return multi_rnn_cell

	def _init_placeholders(self):
		""" Encoder is time-major, Decoder is batch-major """
		self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
		self.encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')

		# required for training, not required for testing
		self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
		self.decoder_targets_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_targets_length')

	def _init_decoder_inputs(self):
		with tf.name_scope('DecoderTrainFeeds'):
			sequence_size, batch_size = tf.unstack(tf.shape(self.decoder_targets))

			EOS = self.embedding.get_token("EOS")
			PAD = self.embedding.get_token("PAD")
			EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * EOS
			PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * PAD

			self.start_token = tf.ones([batch_size], dtype=tf.int32) * EOS
			self.end_token = EOS

			self.decoder_train_inputs = tf.concat([EOS_SLICE, self.decoder_targets], axis=0)
			self.decoder_train_length = self.decoder_targets_length + 1

			decoder_train_targets = tf.concat([self.decoder_targets, PAD_SLICE], axis=0)
			decoder_train_targets_seq_len, _ = tf.unstack(tf.shape(decoder_train_targets))
			decoder_train_targets_eos_mask	 = tf.one_hot(self.decoder_train_length - 1, decoder_train_targets_seq_len, on_value=EOS, off_value=0, dtype=tf.int32)
			decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask, [1, 0])

			# hacky way using one_hot to put EOS symbol at the end of target sequence
			decoder_train_targets = tf.add(decoder_train_targets, decoder_train_targets_eos_mask)

			self.decoder_train_targets = decoder_train_targets

			self.decoder_train_inputs_embedded = self.embedding.lookup(self.decoder_train_inputs)

			self.loss_weights = self._make_loss_weights(batch_size, self.decoder_train_length, name="loss_weight")
			self.batch_size = batch_size

	def _make_loss_weights(self, batch_size, length, name = None):
		"""
			*** hacky way to generate some ones and zeros
			for loss weights used in sequence_loss.

			tensorflow equivalent of [[1] * x + [0] * (length - x) for x in length]
		"""
		max_size = tf.reduce_max(length)
		zeros = tf.zeros([batch_size, max_size], dtype=tf.int32)
		r = tf.range(max_size)
		reshape = tf.reshape(length, [batch_size, 1])
		numbers = r + zeros
		values = reshape + zeros
		bools = tf.less(numbers, values)
		zeros_f = tf.zeros([batch_size, max_size], dtype=tf.float32)
		ones_f = tf.ones([batch_size, max_size], dtype=tf.float32)
		result = tf.where(bools, ones_f, zeros_f, name=name)
		return result

	def _init_encoder_inputs(self):
		with tf.name_scope('EncoderInputs'):
			self.encoder_inputs_embedded = self.embedding.lookup(self.encoder_inputs)

	def _init_encoder(self):
		with tf.name_scope('Encoder'):
			encoder = self.encoder_cell
			zero_state = encoder.zero_state(self.batch_size, dtype=tf.float32)
			outputs, state = tf.nn.dynamic_rnn(encoder, self.encoder_inputs_embedded, initial_state=zero_state, time_major=True, sequence_length=self.encoder_inputs_length)
			self.encoder_outputs, self.encoder_state = outputs, state
			self.encoder_outputs_batch_major = tf.transpose(self.encoder_outputs, [1, 0, 2])

	def _init_decoder(self):
		with tf.name_scope('Decoder'):
			self.attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.encoder_hidden_size, memory_sequence_length=self.encoder_inputs_length, memory=self.encoder_outputs_batch_major)
			self.attention_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(self.decoder_cell, attention_mechanism=self.attention_mechanism, attention_size=self.decoder_hidden_size)
			tmp_state = self.attention_cell.zero_state(self.batch_size, dtype=tf.float32)
			init_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(cell_state=self.encoder_state, attention=tmp_state.attention)
			self.decoder_init_state = init_state

	def _init_decoder_train(self):
		with tf.variable_scope('DecoderHelper') as scope:
			self.decoder_output_layer = tf.contrib.seq2seq.DenseLayer(self.embedding.vocab_size)
			self.train_helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_train_inputs_embedded, self.decoder_train_length, time_major=True)
			self.decoder_train = tf.contrib.seq2seq.BasicDecoder(self.attention_cell, self.train_helper, initial_state=self.decoder_init_state, output_layer=self.decoder_output_layer)
			self.decoder_train_outputs, self.train_final_state = tf.contrib.seq2seq.dynamic_decode(self.decoder_train, output_time_major=True, scope=scope)
			scope.reuse_variables()
			self.inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding.embeddings, self.start_token, self.end_token)
			self.decoder_inference = tf.contrib.seq2seq.BasicDecoder(self.attention_cell, self.inference_helper, initial_state=self.decoder_init_state, output_layer=self.decoder_output_layer)
			self.decoder_inference_outputs, self.inference_final_state = tf.contrib.seq2seq.dynamic_decode(self.decoder_inference, output_time_major=True, scope=scope, maximum_iterations=10)

	def _init_decoder_inference(self):
		return
		with tf.name_scope('DecoderInference') as scope:
			self.train_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding.embeddings, self.start_token, self.end_token)
			self.decoder_inference = tf.contrib.seq2seq.BasicDecoder(self.attention_cell, self.train_helper, initial_state=self.decoder_init_state)
			self.decoder_inference_outputs, self.inference_final_state = tf.contrib.seq2seq.dynamic_decode(self.decoder_inference, output_time_major=True, scope=scope)

	def _init_loss(self):
		logits = tf.transpose(self.decoder_train_outputs.rnn_output, [1, 0, 2])
		targets = tf.transpose(self.decoder_train_targets, [1, 0])
		self.loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=targets, weights=self.loss_weights, name="sequence_loss")
		self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

	def _init_accuracy(self):
		"""Hate you tensorflow"""
		pred_labels = tf.cast(tf.argmax(self.decoder_inference_outputs.rnn_output, axis=2), tf.int32)
		# get shape of pred_labels and train targets
		time1, batch = tf.unstack(tf.shape(pred_labels))
		time2, _ = tf.unstack(tf.shape(self.decoder_train_targets))
		# get minimum sequence size
		time = tf.reduce_min([time1, time2])

		# slice prediction and target to minimum sequence size
		pred_labels_sliced = tf.slice(pred_labels, [0, 0], [time, batch])
		targets_sliced = tf.slice(self.decoder_train_targets, [0, 0], [time, batch])

		# correct label matrix
		correnct_labels = tf.cast(tf.equal(pred_labels_sliced, targets_sliced), tf.float32)

		# set pred labels for logging
		self.pred_labels_sliced = pred_labels_sliced

		# calculate average accuracy
		self.accuracy = tf.reduce_mean(correnct_labels)

	def _pad(self, batch, size, mask):
		return [seq + [mask] * (size - len(seq)) for seq in batch]

	def _time_major(self, batch):
		# print("batch", batch)
		return np.array(batch).transpose([1, 0])

	def get_validation(self, input, output):
		q_validation = [[self.embedding.get_token(token) for token in question] for question in input]
		a_validation = [[self.embedding.get_token(token) for token in answer] for answer in output]

		q_length = [len(q) for q in q_validation]
		a_length = [len(a) for a in a_validation]

		q_max_size = max(q_length)
		a_max_size = max(a_length)

		q_padded = self._pad(q_validation, q_max_size, self.embedding.get_token('PAD'))
		a_padded = self._pad(a_validation, a_max_size, self.embedding.get_token('PAD'))

		return self._time_major(q_padded), self._time_major(a_padded), q_length, a_length

	def get_next_batch(self, input, output, batch_size):
		for i in range(0, len(input), batch_size):
			q_batch = [[self.embedding.get_token(token) for token in question] for question in input[i:i + batch_size]]
			a_batch = [[self.embedding.get_token(token) for token in answer] for answer in output[i:i + batch_size]]

			q_length = [len(q) for q in q_batch]
			a_length = [len(a) for a in a_batch]

			q_max_size = max(q_length)
			a_max_size = max(a_length)

			padded_questions = self._pad(q_batch, q_max_size, self.embedding.get_token('PAD'))
			padded_answers = self._pad(a_batch, a_max_size, self.embedding.get_token('PAD'))

			yield self._time_major(padded_questions), self._time_major(padded_answers), q_length, a_length

	def _dict(self, batch_q, batch_a, batch_q_len, batch_a_len):
		return {self.encoder_inputs: batch_q, self.encoder_inputs_length: batch_q_len, self.decoder_targets: batch_a, self.decoder_targets_length: batch_a_len}

	def train(self, input, output, batch_size=1, num_epochs=1000, count_loss=10):
		# bs = tf.placeholder(shape=(), dtype=tf.int32)
		# lengths = tf.placeholder(shape=(None), dtype=tf.int32)
		saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
			for epoch in range(num_epochs):
				for batch_q, batch_a, batch_q_len, batch_a_len in self.get_next_batch(input, output, batch_size):
					# print(batch_q, batch_q_len)
					# print(batch_a, batch_a_len)
					# dtie = sess.run(self.decoder_train_inputs_embedded, feed_dict=self._dict(batch_q, batch_a, batch_q_len, batch_a_len))
					# print(dtie.shape)
					# return
					_, loss = sess.run([self.train_op, self.loss], feed_dict=self._dict(batch_q, batch_a, batch_q_len, batch_a_len))
					# sys.stdout.write("\r%f" % loss)
					# x = sess.run(self._make_loss_weights(bs, lengths), feed_dict={bs: 10, lengths: [random.randint(1, 9) for i in range(10)]})
					# print(x)
				# print()
				if epoch % count_loss == 0:
					q, a, q_len, a_len = self.get_validation(input, output)
					accuracy, result, targets = sess.run([self.accuracy, self.pred_labels_sliced, self.decoder_train_targets], feed_dict=self._dict(q, a, q_len, a_len))
					print("loss		:\t", loss)
					print("accuracy :\t", accuracy)
					print(result)
					print(a)
					print(targets)
					print()

if __name__ == "__main__":
	print(random.randint(0, 9))
	embedding = emb.Embedding(trainable=True)
	embedding.load("./tmp_vectors.txt")
	embedding.init()
	# make a seq2seq model, with embeddings loaded from tmp_vector_file
	model = Seq2Seq(embedding, 100)
	# 3 questions and answers: 1- Q: salam khoobi  => A: mersi khoobam, 2- Q: che khabar => A: salamati ....
	model.train([["salam", "khoobi"], ["che", "khabar"], ["aya", "hava", "sarde"]], [["mersi", "khoobam"], ["salamati"], ["are", "fekr", "konam"]])
