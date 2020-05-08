class IndexingDataset(Dataset):
	""" A Dataset subclass from which examples include their indices, which
	act like unique IDs---__getitem__(idx) returns (idx, (inputs, labels))."""
	def __init__(self, dataset):
		self.dataset = dataset

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		return idx, self.dataset[idx]
