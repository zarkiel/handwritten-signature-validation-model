import os
import random
import numpy as np
import pandas as pd
import cv2
from preprocessing import preprocess_image

class SignatureDataGenerator:
    def __init__(self, dataset, pairs_file='signature_pairs.csv',
                 seed=42, epsilon=1e-5):
        self.pairs_file = pairs_file
        self.seed = seed
        self.epsilon = epsilon
        self.pairs_df = pd.read_csv(os.path.join(pairs_file))
        self.cedar1_pairs()
        
    def cedar1_pairs(self):
        self.lines_per_signer = 852
        self.total_signers = 55

        total_rows = len(self.pairs_df)
        max_possible_signers = total_rows // self.lines_per_signer
        if max_possible_signers < self.total_signers:
            self.total_signers = max_possible_signers

        all_signers = np.arange(self.total_signers)

        np.random.seed(self.seed)

        train_signers = np.random.choice(all_signers, size=45, replace=False)
        remaining_signers = np.setdiff1d(all_signers, train_signers)
        val_signers = np.random.choice(remaining_signers, size=5, replace=False)
        test_signers = np.setdiff1d(remaining_signers, val_signers)

        def get_indices(signers):
            indices = []
            for s in signers:
                start = s * self.lines_per_signer
                end = start + self.lines_per_signer
                indices.extend(range(start, end))
            return indices

        train_indices = get_indices(train_signers)
        val_indices = get_indices(val_signers)
        test_indices = get_indices(test_signers)

        self.train_pairs = self.pairs_df.iloc[train_indices].reset_index(drop=True).values.tolist()
        self.val_pairs = self.pairs_df.iloc[val_indices].reset_index(drop=True).values.tolist()
        self.test_pairs = self.pairs_df.iloc[test_indices].reset_index(drop=True).values.tolist()

    def _load_image(self, path):
        full_path = os.path.join(path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")

        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        img = preprocess_image(img)

        return img

    def _generate_pairs(self, pairs, batch_size):
        while True:
            batch_images1, batch_images2, batch_labels = [], [], []

            positive_pairs = [pair for pair in pairs if int(pair[2]) == 1]
            negative_pairs = [pair for pair in pairs if int(pair[2]) == 0]

            random.shuffle(positive_pairs)
            random.shuffle(negative_pairs)

            half_batch = batch_size // 2

            selected_pairs = positive_pairs[:half_batch] + negative_pairs[:half_batch]
            random.shuffle(selected_pairs)

            for file1, file2, label in selected_pairs:
                try:
                    img1 = self._load_image(file1)
                    img2 = self._load_image(file2)

                    batch_images1.append(img1)
                    batch_images2.append(img2)
                    batch_labels.append(int(label))
                except Exception as e:
                    print(f"Error processing pair ({file1}, {file2}): {e}")
                    continue

            yield (np.array(batch_images1), np.array(batch_images2)), np.array(batch_labels)

    def train_generator(self, batch_size):
        return self._generate_pairs(self.train_pairs, batch_size)

    def validation_generator(self, batch_size):
        return self._generate_pairs(self.val_pairs, batch_size)

    def test_generator(self, batch_size):
        return self._generate_pairs(self.test_pairs, batch_size)
