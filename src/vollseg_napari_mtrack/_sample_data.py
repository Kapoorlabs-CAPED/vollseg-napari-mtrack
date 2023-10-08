from ._reader import reader_function
from vollseg import test_microtubule_kymographs


def get_microtubule_test_data():

    extracted_folder = test_microtubule_kymographs()
    image = reader_function(extracted_folder)[0][0]
    return [(image, {"name": "microtubule_kymographs"})]
