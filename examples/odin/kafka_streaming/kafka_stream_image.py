from kafka import KafkaProducer
import os
from PIL import Image
import numpy as np
from streaming_data_types.area_detector_ADAr import serialise_ADAr, \
    deserialise_ADAr
from datetime import datetime
import time

if __name__ == '__main__':
    producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                             api_version=(0, 10, 1))
    kafka_producer = KafkaProducer(bootstrap_servers='localhost:9092')
    idx = 150
    for i in range(1, 201):
        idx += 1
        nbr_str = str(i)
        nbr_zeros = 4 - len(nbr_str)
        first_zeros = '0' * nbr_zeros
        path_to_image = os.path.join('Lego1',
                                     f'tomo_{first_zeros}{nbr_str}.tif')
        image = Image.open(path_to_image)
        img_array = np.array(image)
        serialized_output = serialise_ADAr('image_source', idx, datetime.now(),
                                                          img_array)
        kafka_producer.send('odin_topic', serialized_output)
        time.sleep(1)

