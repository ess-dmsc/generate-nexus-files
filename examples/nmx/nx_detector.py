import itertools
from typing import List

try:
    from nx_component import NXComponent
except ModuleNotFoundError:
    from examples.nmx.nx_component import NXComponent


class NXDetector(NXComponent):
    def __init__(self, name: str, instrument_name: str):
        super().__init__(name)
        self.instrument_name = instrument_name

    def get_detector_numbers(self):
        raise NotImplementedError

    def get_x_pixel_offsets(self):
        raise NotImplementedError

    def get_y_pixel_offsets(self):
        raise NotImplementedError

    def get_z_pixel_offsets(self):
        raise NotImplementedError


class BoxNXDetector(NXDetector):
    def __init__(
        self,
        name: str,
        instrument_name: str,
        number_of_pixels_x: int,
        number_of_pixels_y: int,
        size_z: float,
        channel_pitch_x: float,
        channel_pitch_y: float,
        first_pixel_id: int = 1,
        gap_every_x_pixels: int = 0,
        gap_every_y_pixels: int = 0,
        gap_width_x: float = 0,
        gap_width_y: float = 0,
    ):
        super().__init__(name, instrument_name)
        self.number_of_pixels_x = number_of_pixels_x
        self.number_of_pixels_y = number_of_pixels_y
        self.channel_pitch_x = channel_pitch_x
        self.channel_pitch_y = channel_pitch_y
        self.first_pixel_id = first_pixel_id
        self.gap_every_x_pixels = gap_every_x_pixels
        self.gap_every_y_pixels = gap_every_y_pixels
        self.gap_width_x = gap_width_x
        self.gap_width_y = gap_width_y
        self.size_x = (
            number_of_pixels_x * channel_pitch_x
            + ((number_of_pixels_x - 1) // gap_every_x_pixels) * gap_width_x
            if gap_every_x_pixels != 0
            else number_of_pixels_x * channel_pitch_x
        )
        self.size_y = (
            number_of_pixels_y * channel_pitch_y
            + ((number_of_pixels_y - 1) // gap_every_y_pixels) * gap_width_y
            if gap_every_y_pixels != 0
            else number_of_pixels_y * channel_pitch_y
        )
        self.size_z = size_z

    def get_pixel_coordinates(self, pixel_id: int) -> List[float]:
        assert self.is_valid_pixel_id(
            pixel_id
        ), f"Pixel {pixel_id} out of bounds (first={self.first_pixel_id} last={self.last_pixel_id})"
        pixel_id -= self.first_pixel_id
        column = pixel_id % self.number_of_pixels_x
        row = pixel_id // self.number_of_pixels_x
        gaps_x = (
            column // self.gap_every_x_pixels if self.gap_every_x_pixels != 0 else 0
        )
        gaps_y = row // self.gap_every_y_pixels if self.gap_every_y_pixels != 0 else 0
        x = (
            column * self.channel_pitch_x
            + gaps_x * self.gap_width_x
            - self.size_x / 2
            + self.channel_pitch_x / 2
        )
        y = self.size_y / 2 - (
            row * self.channel_pitch_y
            + gaps_y * self.gap_width_y
            + self.channel_pitch_y / 2
        )
        z = 0
        return [x, y, z]

    @property
    def number_of_pixels(self):
        return self.number_of_pixels_x * self.number_of_pixels_y

    @property
    def last_pixel_id(self):
        return self.first_pixel_id + self.number_of_pixels - 1

    @property
    def pixel_ids(self):
        for pixel_id in range(
            self.first_pixel_id,
            self.first_pixel_id + (self.number_of_pixels_x * self.number_of_pixels_y),
        ):
            yield pixel_id

    def get_detector_numbers(self):
        """Generator of detector (pixel) numbers. Returns one detector row at a time."""
        pixel_ids_generator = self.pixel_ids
        while True:
            slice_gen = list(
                itertools.islice(pixel_ids_generator, self.number_of_pixels_x)
            )
            if not slice_gen:
                break
            yield slice_gen

    def get_x_pixel_offsets(self):
        """Return list of x pixel offsets for the first row of the detector.
        Assumes that all rows have identical offsets."""
        row = next(self.get_detector_numbers(), [])
        offsets = []
        for pixel in row:
            x, _, _ = self.get_pixel_coordinates(pixel)
            offsets.append(x)
        return offsets

    def get_y_pixel_offsets(self):
        """Return list of y pixel offsets for the first column of the detector.
        Assumes that all columns have identical offsets."""
        offsets = []
        for row in self.get_detector_numbers():
            _, y, _ = self.get_pixel_coordinates(row[0])
            offsets.append(y)
        return offsets

    def get_z_pixel_offsets(self):
        """Return a list of z pixel offsets, with a single item of value zero."""
        return [0]

    def is_valid_pixel_id(self, pixel_id: int) -> bool:
        """
        Check if a given pixel ID is within the range of the detector.
        """
        return (
            self.first_pixel_id
            <= pixel_id
            < self.first_pixel_id + self.number_of_pixels
        )

    def to_json(self):
        context = {
            "j2_detector_name": self.name,
            "j2_instrument_name": self.instrument_name,
            "j2_detector_sizes": [self.size_z, self.size_y, self.size_x],
            "j2_detector_numbers": list(self.get_detector_numbers()),
            "j2_x_pixel_offsets": self.get_x_pixel_offsets(),
            "j2_y_pixel_offsets": self.get_y_pixel_offsets(),
            "j2_z_pixel_offsets": self.get_z_pixel_offsets(),
            "j2_detector_transformations": self.transformations,
        }
        return self._render(
            "examples/nmx", "template_NXdetector_box.json.j2", **context
        )
