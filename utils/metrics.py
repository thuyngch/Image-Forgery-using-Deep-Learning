#------------------------------------------------------------------------------
#	Import libraries
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#	Binary classification metrics
#------------------------------------------------------------------------------
class BinaryClassificationMetrics(object):
	def __init__(self, logging=None):
		super(BinaryClassificationMetrics, self).__init__()	
		self.logging = logging

	def basics(self, scores_pos, scores_neg):
		N_p = len(scores_pos)
		self.TP = sum(scores_pos)
		self.FP = (N_p - self.TP)

		N_n = len(scores_neg)
		self.FN = sum(scores_neg)
		self.TN = (N_n - self.FN)
		return self.TP/N_p, self.FP/N_p, self.TN/N_n, self.FN/N_n

	def accuracy(self, scores_pos, scores_neg):
		N_p = len(scores_pos)
		TP = sum(scores_pos)
		N_n = len(scores_neg)
		FN = sum(scores_neg)
		TN = N_n - FN
		self.acc = (TP + TN) / (N_p + N_n)
		return self.acc

	def precision(self, scores_pos, scores_neg):
		N_p = len(scores_pos)
		TP = sum(scores_pos)
		FP = N_p - TP
		self.pre = TP / (TP + FP)
		return self.pre

	def recall(self, scores_pos, scores_neg):
		TP = sum(scores_pos)
		FN = sum(scores_neg)
		self.rec = TP / (TP + FN)
		return self.rec

	def fscore(self, scores_pos, scores_neg):
		pre = self.precision(scores_pos, scores_neg)
		rec = self.recall(scores_pos, scores_neg)
		self.fs = 2*pre*rec / (pre + rec)
		return self.fs

	def compute_all(self, scores_pos, scores_neg):
		self.basics(scores_pos, scores_neg)
		self.accuracy(scores_pos, scores_neg)
		self.precision(scores_pos, scores_neg)
		self.recall(scores_pos, scores_neg)
		self.fscore(scores_pos, scores_neg)

	def print_metrics(self):
		TP, FP = self.TP/(self.TP+self.FP), self.FP/(self.TP+self.FP)
		TN, FN = self.TN/(self.TN+self.FN), self.FN/(self.TN+self.FN)
		printf = self.logging.info if self.logging else print
		printf("TP = %.2f %%; FP = %.2f %%" % (TP*100, FP*100))
		printf("TN = %.2f %%; FN = %.2f %%" % (TN*100, FN*100))
		printf("Accuracy = %.2f %%" % (100*self.acc))
		printf("Precision = %.2f %%" % (100*self.pre))
		printf("Recall = %.2f %%" % (100*self.rec))
		printf("F-score = %.2f %%" % (100*self.fs))

	def write_to_file(self, file):
		fp = open(file, "w")
		TP, FP = self.TP/(self.TP+self.FP), self.FP/(self.TP+self.FP)
		TN, FN = self.TN/(self.TN+self.FN), self.FN/(self.TN+self.FN)
		fp.write("TP = %.2f %%; FP = %.2f %%\n" % (TP*100, FP*100))
		fp.write("TN = %.2f %%; FN = %.2f %%\n" % (TN*100, FN*100))
		fp.write("Accuracy = %.2f %%\n" % (100*self.acc))
		fp.write("Precision = %.2f %%\n" % (100*self.pre))
		fp.write("Recall = %.2f %%\n" % (100*self.rec))
		fp.write("F-score = %.2f %%\n" % (100*self.fs))
		fp.close()