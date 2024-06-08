import unittest
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from pymongo import MongoClient
import os

class TestDataAugmentation(unittest.TestCase):

    def test_data_augmentation(self):
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Ustvarimo vzorčno sliko (128x128 z enim kanalom)
        sample_image = np.random.rand(128, 128, 1)
        sample_image = np.expand_dims(sample_image, 0)  # Razširimo dimenzije, da ustrezajo pričakovani vhodni obliki

        # Ustvarimo iterator za augmentacijo podatkov
        augmented_images = datagen.flow(sample_image, batch_size=1)

        # Preverimo, ali so slike pravilno augmentirane
        for i in range(5):
            augmented_image = augmented_images.next()
            self.assertEqual(augmented_image.shape, (1, 128, 128, 1))

class TestMongoDBConnection(unittest.TestCase):

    def setUp(self):
        # Pridobite MongoDB URL iz okoljske spremenljivke ali uporabite privzeto vrednost
        self.mongo_url = os.getenv('MONGO_URL', 'mongodb://localhost:27017')
        self.client = MongoClient(self.mongo_url)
        self.db = self.client['pametni-paketnik']
        self.users_collection = self.db['users']

    def tearDown(self):
        # Zaprite povezavo s podatkovno bazo
        self.client.close()

    def test_connection(self):
        # Preverite, ali je povezava vzpostavljena
        self.assertTrue(self.client is not None)

    def test_find_user(self):
        # Preverite, ali lahko najdete uporabnika (če obstaja)
        user = self.users_collection.find_one()
        self.assertTrue(user is not None)

if __name__ == '__main__':
    unittest.main()
