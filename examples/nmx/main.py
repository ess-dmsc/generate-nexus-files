import json
import jinja2

try:
    from build_nmx_json import BoxNXDetector
except ModuleNotFoundError:
    from examples.nmx.build_nmx_json import BoxNXDetector


TEMPLATES_DIR = "examples/nmx"
TEMPLATE_NAME = "template_nmx_v3.0_baseline.json.j2"
OUTPUT_FILE_NAME = "nmx_v3.0_small.json"
OUTPUT_FILE_NAME = "nmx_v3.0_compact.json"

FACTOR = 80  # used to reduce file size while testing. Set to 1 for actual numbers.
PIXELS_X = int(1280 / FACTOR)  # actual=1280
PIXELS_Y = int(1280 / FACTOR)  # actual=1280
CHANNEL_PITCH_X = 0.4 * FACTOR  # actual=0.4mm
CHANNEL_PITCH_Y = CHANNEL_PITCH_X
DEPTH_OF_DETECTOR = 10.0  # actual=unknown
GAP_EVERY_X_PIXELS = int(640 / FACTOR)
GAP_EVERY_Y_PIXELS = int(GAP_EVERY_X_PIXELS)
GAP_WIDTH_X = 0.2  # actual=0.2mm
GAP_WIDTH_Y = GAP_WIDTH_X


def render_template(template_dir, template_file_name, **context):
    return (
        jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir + "/"))
        .get_template(template_file_name)
        .render(context)
    )


def is_valid_json(file_path):
    try:
        with open(file_path, "r") as f:
            json.load(f)
        return True, None
    except json.JSONDecodeError as exc:
        return False, exc


def render():
    print(f"Creating detectors...")
    detector_panel_0 = BoxNXDetector(
        "detector_panel_0",
        number_of_pixels_x=PIXELS_X,
        number_of_pixels_y=PIXELS_Y,
        size_z=DEPTH_OF_DETECTOR,
        channel_pitch_x=CHANNEL_PITCH_X,
        channel_pitch_y=CHANNEL_PITCH_Y,
        first_pixel_id=1,
        gap_every_x_pixels=GAP_EVERY_X_PIXELS,
        gap_every_y_pixels=GAP_EVERY_Y_PIXELS,
        gap_width_x=GAP_WIDTH_X,
        gap_width_y=GAP_WIDTH_Y,
    )
    (
        detector_panel_0.translate("x_offset_sample_to_stageZ", x=-0.4)
        .translate("y_offset_sample_to_stageZ", y=-0.7)
        .translate("stageZ", z=1.1)
        .translate("y_offset_stageZ_to_axis1", y=0.1)
        .rotate("axis1", y=90)
        .translate("x_offset_axis1_to_axis2", x=0.12)
        .translate("y_offset_axis1_to_axis2", x=0.1)
        .rotate("axis2", z=-45)
        .translate("y_offset_axis2_to_axis3", y=0.4)
        .rotate("axis3", z=45)
        .translate("x_offset_axis3_to_axis4", x=0.05)
        .translate("y_offset_axis3_to_axis4", y=0.12)
        .translate("z_offset_axis3_to_axis4", z=0.06)
        .rotate("axis4", x=0)
        .translate("x_offset_axis4_to_axis5", x=0.3)
        .rotate("axis5", z=0)
        .translate("x_offset_axis5_to_axis6", x=0.15)
        .rotate("axis6", x=0)
    )

    detector_panel_1 = BoxNXDetector(
        "detector_panel_1",
        number_of_pixels_x=PIXELS_X,
        number_of_pixels_y=PIXELS_Y,
        size_z=DEPTH_OF_DETECTOR,
        channel_pitch_x=CHANNEL_PITCH_X,
        channel_pitch_y=CHANNEL_PITCH_Y,
        first_pixel_id=1 + (PIXELS_X * PIXELS_Y),
        gap_every_x_pixels=GAP_EVERY_X_PIXELS,
        gap_every_y_pixels=GAP_EVERY_Y_PIXELS,
        gap_width_x=GAP_WIDTH_X,
        gap_width_y=GAP_WIDTH_Y,
    )
    (
        detector_panel_1.translate("x_offset_sample_to_stageZ", x=-0.1)
        .translate("y_offset_sample_to_stageZ", y=0)
        .translate("stageZ", z=1)
        .translate("y_offset_stageZ_to_axis1", y=0.1)
        .rotate("axis1", y=0)
        .translate("x_offset_axis1_to_axis2", x=0.12)
        .translate("y_offset_axis1_to_axis2", x=0.1)
        .rotate("axis2", z=-45)
        .translate("y_offset_axis2_to_axis3", y=0.4)
        .rotate("axis3", z=-40)
        .translate("x_offset_axis3_to_axis4", x=0.05)
        .translate("y_offset_axis3_to_axis4", y=0.12)
        .translate("z_offset_axis3_to_axis4", z=0.06)
        .rotate("axis4", x=0)
        .translate("x_offset_axis4_to_axis5", x=0.3)
        .rotate("axis5", z=0)
        .translate("x_offset_axis5_to_axis6", x=0.15)
        .rotate("axis6", x=0)
    )


    detector_panel_2 = BoxNXDetector(
        "detector_panel_2",
        number_of_pixels_x=PIXELS_X,
        number_of_pixels_y=PIXELS_Y,
        size_z=DEPTH_OF_DETECTOR,
        channel_pitch_x=CHANNEL_PITCH_X,
        channel_pitch_y=CHANNEL_PITCH_Y,
        first_pixel_id=1 + (2 * PIXELS_X * PIXELS_Y),
        gap_every_x_pixels=GAP_EVERY_X_PIXELS,
        gap_every_y_pixels=GAP_EVERY_Y_PIXELS,
        gap_width_x=GAP_WIDTH_X,
        gap_width_y=GAP_WIDTH_Y,
    )
    (
        detector_panel_2.translate("x_offset_sample_to_stageZ", x=0.4)
        .translate("y_offset_sample_to_stageZ", y=0)
        .translate("stageZ", z=-0.5)
        .rotate("rotate_arm_chair", y=180)
        .translate("y_offset_stageZ_to_axis1", y=0.1)
        .rotate("axis1", y=90)
        .translate("x_offset_axis1_to_axis2", x=0.12)
        .translate("y_offset_axis1_to_axis2", x=0.1)
        .rotate("axis2", z=0)
        .translate("y_offset_axis2_to_axis3", y=0.4)
        .rotate("axis3", z=0)
        .translate("x_offset_axis3_to_axis4", x=0.05)
        .translate("y_offset_axis3_to_axis4", y=0.12)
        .translate("z_offset_axis3_to_axis4", z=0.06)
        .rotate("axis4", x=90)
        .translate("x_offset_axis4_to_axis5", x=0.3)
        .rotate("axis5", z=0)
        .translate("x_offset_axis5_to_axis6", x=0.15)
        .rotate("axis6", x=0)
    )

    print(f"Rendering detectors...")
    context = {
        "j2_instrument_detector_panel_0": detector_panel_0.to_json(),
        "j2_instrument_detector_panel_1": detector_panel_1.to_json(),
        "j2_instrument_detector_panel_2": detector_panel_2.to_json(),
    }

    print(f"Rendering template...")
    return render_template(TEMPLATES_DIR, TEMPLATE_NAME, **context)


def main():
    # detector = BoxNXDetector(
    #     "test_detector",
    #     number_of_pixels_x=2,
    #     number_of_pixels_y=2,
    #     size_z=10.0,
    #     channel_pitch_x=128,
    #     channel_pitch_y=128,
    #     first_pixel_id=1,  # Set the first pixel to start at 1
    # )
    # with open("output-1.json", "w", encoding="utf-8") as file:
    #     file.write(detector.to_json())

    # detector = BoxNXDetector(
    #     "test_detector",
    #     number_of_pixels_x=2,
    #     number_of_pixels_y=2,
    #     size_z=10.0,
    #     channel_pitch_x=128,
    #     channel_pitch_y=128,
    #     first_pixel_id=1,  # Set the first pixel to start at 1
    #     gap_every_x_pixels=1,
    #     gap_every_y_pixels=1,
    #     gap_width_x=10,
    #     gap_width_y=10,
    # )
    # with open("output-2.json", "w", encoding="utf-8") as file:
    #     file.write(detector.to_json())

    output = render()
    with open(OUTPUT_FILE_NAME, "w", encoding="utf-8") as file:
        file.write(output)
    is_valid, exc = is_valid_json(OUTPUT_FILE_NAME)
    if is_valid:
        print(f"File passed JSON validation: {OUTPUT_FILE_NAME}")
    else:
        print(f"JSON validation failed: {exc}")
        raise exc


if __name__ == "__main__":
    main()
