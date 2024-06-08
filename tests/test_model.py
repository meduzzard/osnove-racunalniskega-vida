import unittest
import pymongo
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

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
        # Povezava z MongoDB
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["testdb"]
        self.collection = self.db["users"]

        # Ustvarimo testnega uporabnika
        self.test_user = {"username": "testuser", "email": "testuser@example.com"}
        self.collection.insert_one(self.test_user)

    def tearDown(self):
        # Počistimo podatkovno bazo po vsakem testu
        self.collection.delete_many({})
        self.client.close()

    def test_find_user(self):
        user = self.collection.find_one({"username": "testuser"})
        self.assertTrue(user is not None)

if __name__ == '__main__':
    unittest.main()
