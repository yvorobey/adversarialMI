from abc import abstractmethod
import numpy


class Feature(object):
    @abstractmethod
    def range_check(self):
        pass

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __iadd__(self, other):
        pass

    @abstractmethod
    def __sub__(self, other):
        pass

    @abstractmethod
    def normalize(self, length, norm):
        pass


class RawImage(Feature):
    def __init__(self, _raw_image):
        self.raw_image = _raw_image
        self.with_mas = False
        self.max_value = numpy.max(_raw_image)
        self.min_value = numpy.min(_raw_image)

    def __add__(self, other):
        new = self.raw_image + other.raw_image
        return RawImage(new)

    def __iadd__(self, modification):
        self.raw_image += modification.raw_image
        return self

    def __sub__(self, other):
        difference = self.raw_image - other.raw_image
        return RawImage(difference)

    def copy(self):
        new_data = self.raw_image.copy()
        return RawImage(new_data)

    def range_check(self):
        self.raw_image[self.raw_image > self.max_value] = self.max_value
        self.raw_image[self.raw_image < self.min_value] = self.min_value

    def normalize(self, length, norm=numpy.sign):
        self.raw_image = norm(self.raw_image) * length


class ImageWithMas(Feature):
    def __init__(self, _raw_image, _mas):
        self.raw_image = _raw_image
        self.mas = _mas
        self.with_mas = True
        self.max_value = numpy.max(_raw_image)
        self.min_value = numpy.min(_raw_image)

    def __add__(self, other):
        # assert self.mas is other.mas
        new_raw = self.raw_image + other.raw_image
        return ImageWithMas(new_raw, self.mas)

    def __iadd__(self, modification):
        # assert self.mas is modification.mas
        self.raw_image += modification.raw_image
        return self

    def __sub__(self, other):
        assert self.mas is other.mas
        difference = self.raw_image - other.raw_image
        return ImageWithMas(difference, self.mas)

    def copy(self):
        new_raw_image = self.raw_image.copy()
        mas = self.mas
        return ImageWithMas(new_raw_image, mas)

    def range_check(self):
        self.raw_image[self.raw_image > self.max_value] = self.max_value
        self.raw_image[self.raw_image < self.min_value] = self.min_value

    def normalize(self, length, norm=numpy.sign):
        self.raw_image = norm(self.raw_image) * length