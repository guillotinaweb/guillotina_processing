import numpy as np
import keras
from guillotina.utils import get_object_by_oid
from guillotina_processing.interfaces import ITextExtractor
from guillotina_processing.interfaces import ILabelExtractor
from guillotina_processing.utils import cleanup_text


class DataTextGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(
            self,
            list_IDs,
            all_labels,
            vocabulary,
            batch_size=32,
            dim=(32, 32, 32),
            n_classes=10,
            shuffle=True,
            timeout=30,
            loop=None):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.timeout = timeout
        self.loop = loop
        self.vocabulary = vocabulary
        self.all_labels = all_labels
        self.labels = []
        self.list_IDs = list_IDs
        self.n_classes = len(self.all_labels)
        self.current_index = 0
        self.dim = dim
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        self.current_index = 0
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    async def async_next(self):
        indexes = self.indexes[self.current_index*self.batch_size:(self.current_index+1)*self.batch_size]  # noqa
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = await self.__data_generation(list_IDs_temp)
        self.current_index += 1
        return X, y

    async def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.zeros((self.batch_size, self.dim))
        y = np.zeros((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            obj = await get_object_by_oid(ID)
            # get Text
            text = await ITextExtractor(obj)()
            text = cleanup_text(text)
            for position, word in enumerate(text):
                if position >= self.dim:
                    break
                if word in self.vocabulary.dictionary:
                    X[i, position] = self.vocabulary.dictionary[word]
                else:
                    X[i, position] = self.vocabulary.dictionary["<PAD>"]

            # Store class
            label = await ILabelExtractor(obj)()
            if label not in self.labels:
                self.labels.append(label)
            index_temp = self.labels.index(label)
            y[i] = 0 if index_temp > 1 else index_temp

        X = keras.preprocessing.sequence.pad_sequences(
            X,
            value=self.vocabulary.dictionary["<PAD>"],
            padding='post',
            maxlen=self.dim)
        category = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, category
