import argparse
import json
import os

import jinja2

try:
    from nx_detector import BoxNXDetector
except ModuleNotFoundError:
    from examples.nmx.nx_detector import BoxNXDetector


FACTOR = 1  # used to reduce file size while testing. Set to 1 for actual numbers.
# FACTOR = 80  # used to reduce file size while testing. Set to 1 for actual numbers.
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


def render(template_dir, template_file_name):
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
    # Example of manual positioning instead of using NXlog:
    # (
    #     detector_panel_0.translate("x_offset_sample_to_stageZ", x=-0.4)
    #     .translate("y_offset_sample_to_stageZ", y=-0.7)
    #     .translate("stageZ", z=0.8)
    #     .translate("y_offset_stageZ_to_axis1", y=0.1)
    #     .rotate("axis1", y=90)
    #     .translate("x_offset_axis1_to_axis2", x=0.12)
    #     .translate("y_offset_axis1_to_axis2", x=0.1)
    #     .rotate("axis2", z=-45)
    #     .translate("y_offset_axis2_to_axis3", y=0.4)
    #     .rotate("axis3", z=45)
    #     .translate("x_offset_axis3_to_axis4", x=0.05)
    #     .translate("y_offset_axis3_to_axis4", y=0.12)
    #     .translate("z_offset_axis3_to_axis4", z=0.06)
    #     .rotate("axis4", x=0)
    #     .translate("x_offset_axis4_to_axis5", x=0.3)
    #     .rotate("axis5", z=0)
    #     .translate("x_offset_axis5_to_axis6", x=0.15)
    #     .rotate("axis6", x=0)
    # )
    (
        detector_panel_0
        .translate("x_offset_sample_to_stageZ", x=-0.8)
        .translate("y_offset_sample_to_stageZ", y=-0.7)
        .translate("z_offset_sample_to_stageZ", z=0.1)
        .translate_from_nxlog("stageZ", [0,0,1], "f144", "nmx_robot", "NMX:Robot0:stageZ")
        .translate("y_offset_stageZ_to_axis1", y=0.1)
        .rotate_from_nxlog("axis1", [0,1,0], "f144", "nmx_robot", "NMX:Robot0:axis1")
        .translate("x_offset_axis1_to_axis2", x=0.12)
        .translate("y_offset_axis1_to_axis2", x=0.1)
        .rotate_from_nxlog("axis2", [0,0,1], "f144", "nmx_robot", "NMX:Robot0:axis2")
        .translate("y_offset_axis2_to_axis3", y=0.4)
        .rotate_from_nxlog("axis3", [0,0,1], "f144", "nmx_robot", "NMX:Robot0:axis3")
        .translate("x_offset_axis3_to_axis4", x=0.05)
        .translate("y_offset_axis3_to_axis4", y=0.12)
        .translate("z_offset_axis3_to_axis4", z=0.06)
        .rotate_from_nxlog("axis4", [1,0,0], "f144", "nmx_robot", "NMX:Robot0:axis4")
        .translate("x_offset_axis4_to_axis5", x=0.3)
        .rotate_from_nxlog("axis5", [0,0,1], "f144", "nmx_robot", "NMX:Robot0:axis5")
        .translate("x_offset_axis5_to_axis6", x=0.15)
        .rotate_from_nxlog("axis6", [1,0,0], "f144", "nmx_robot", "NMX:Robot0:axis6")
        .rotate("orientation", y=90)
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
    # Example of manual positioning instead of using NXlog:
    # (
    #     detector_panel_1.translate("x_offset_sample_to_stageZ", x=-0.9)
    #     .translate("y_offset_sample_to_stageZ", y=0)
    #     .translate("stageZ", z=0.3)
    #     .translate("y_offset_stageZ_to_axis1", y=0.1)
    #     .rotate("axis1", y=0)
    #     .translate("x_offset_axis1_to_axis2", x=0.12)
    #     .translate("y_offset_axis1_to_axis2", x=0.1)
    #     .rotate("axis2", z=-45)
    #     .translate("y_offset_axis2_to_axis3", y=0.4)
    #     .rotate("axis3", z=-40)
    #     .translate("x_offset_axis3_to_axis4", x=0.05)
    #     .translate("y_offset_axis3_to_axis4", y=0.12)
    #     .translate("z_offset_axis3_to_axis4", z=0.06)
    #     .rotate("axis4", x=0)
    #     .translate("x_offset_axis4_to_axis5", x=0.3)
    #     .rotate("axis5", z=0)
    #     .translate("x_offset_axis5_to_axis6", x=0.15)
    #     .rotate("axis6", x=0)
    # )
    (
        detector_panel_1
        .translate("x_offset_sample_to_stageZ", x=-0.8)
        .translate("y_offset_sample_to_stageZ", y=-0.7)
        .translate("z_offset_sample_to_stageZ", z=-0.7)
        .translate_from_nxlog("stageZ", [0,0,1], "f144", "nmx_robot", "NMX:Robot1:stageZ")
        .translate("y_offset_stageZ_to_axis1", y=0.1)
        .rotate_from_nxlog("axis1", [0,1,0], "f144", "nmx_robot", "NMX:Robot1:axis1")
        .translate("x_offset_axis1_to_axis2", x=0.12)
        .translate("y_offset_axis1_to_axis2", x=0.1)
        .rotate_from_nxlog("axis2", [0,0,1], "f144", "nmx_robot", "NMX:Robot1:axis2")
        .translate("y_offset_axis2_to_axis3", y=0.4)
        .rotate_from_nxlog("axis3", [0,0,1], "f144", "nmx_robot", "NMX:Robot1:axis3")
        .translate("x_offset_axis3_to_axis4", x=0.05)
        .translate("y_offset_axis3_to_axis4", y=0.12)
        .translate("z_offset_axis3_to_axis4", z=0.06)
        .rotate_from_nxlog("axis4", [1,0,0], "f144", "nmx_robot", "NMX:Robot1:axis4")
        .translate("x_offset_axis4_to_axis5", x=0.3)
        .rotate_from_nxlog("axis5", [0,0,1], "f144", "nmx_robot", "NMX:Robot1:axis5")
        .translate("x_offset_axis5_to_axis6", x=0.15)
        .rotate_from_nxlog("axis6", [1,0,0], "f144", "nmx_robot", "NMX:Robot1:axis6")
        .rotate("orientation", y=90)
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
    # Example of manual positioning instead of using NXlog:
    # (
    #     detector_panel_2.translate("x_offset_sample_to_stageZ", x=0.4)
    #     .translate("y_offset_sample_to_stageZ", y=-0.2)
    #     .translate("stageZ", z=-0.5)
    #     .rotate("rotate_arm_chair", y=180)
    #     .translate("y_offset_stageZ_to_axis1", y=0.1)
    #     .rotate("axis1", y=90)
    #     .translate("x_offset_axis1_to_axis2", x=0.12)
    #     .translate("y_offset_axis1_to_axis2", x=0.1)
    #     .rotate("axis2", z=0)
    #     .translate("y_offset_axis2_to_axis3", y=0.4)
    #     .rotate("axis3", z=0)
    #     .translate("x_offset_axis3_to_axis4", x=0.05)
    #     .translate("y_offset_axis3_to_axis4", y=0.12)
    #     .translate("z_offset_axis3_to_axis4", z=0.06)
    #     .rotate("axis4", x=90)
    #     .translate("x_offset_axis4_to_axis5", x=0.3)
    #     .rotate("axis5", z=0)
    #     .translate("x_offset_axis5_to_axis6", x=0.15)
    #     .rotate("axis6", x=0)
    # )
    (
        detector_panel_2
        .translate("x_offset_sample_to_stageZ", x=0.8)
        .translate("y_offset_sample_to_stageZ", y=-0.7)
        .translate("z_offset_sample_to_stageZ", z=-0.6)
        .translate_from_nxlog("stageZ", [0,0,1], "f144", "nmx_robot", "NMX:Robot2:stageZ")
        .rotate("arm_chair", y=180)
        .translate("y_offset_stageZ_to_axis1", y=0.1)
        .rotate_from_nxlog("axis1", [0,1,0], "f144", "nmx_robot", "NMX:Robot2:axis1")
        .translate("x_offset_axis1_to_axis2", x=0.12)
        .translate("y_offset_axis1_to_axis2", x=0.1)
        .rotate_from_nxlog("axis2", [0,0,1], "f144", "nmx_robot", "NMX:Robot2:axis2")
        .translate("y_offset_axis2_to_axis3", y=0.4)
        .rotate_from_nxlog("axis3", [0,0,1], "f144", "nmx_robot", "NMX:Robot2:axis3")
        .translate("x_offset_axis3_to_axis4", x=0.05)
        .translate("y_offset_axis3_to_axis4", y=0.12)
        .translate("z_offset_axis3_to_axis4", z=0.06)
        .rotate_from_nxlog("axis4", [1,0,0], "f144", "nmx_robot", "NMX:Robot2:axis4")
        .translate("x_offset_axis4_to_axis5", x=0.3)
        .rotate_from_nxlog("axis5", [0,0,1], "f144", "nmx_robot", "NMX:Robot2:axis5")
        .translate("x_offset_axis5_to_axis6", x=0.15)
        .rotate_from_nxlog("axis6", [1,0,0], "f144", "nmx_robot", "NMX:Robot2:axis6")
        .rotate("orientation", y=-90)
    )

    print(f"Rendering detectors...")
    context = {
        "j2_instrument_detector_panel_0": detector_panel_0.to_json(),
        "j2_instrument_detector_panel_1": detector_panel_1.to_json(),
        "j2_instrument_detector_panel_2": detector_panel_2.to_json(),
    }

    print(f"Rendering template...")
    output = render_template(template_dir, template_file_name, **context)
    json.loads(output)  # raises json.JSONDecodeError if invalid JSON
    return output


def main():
    parser = argparse.ArgumentParser(description="Process output file name.")
    parser.add_argument(
        "-t",
        "--template-file",
        required=True,
        help="Baseline jinja2 template for the instrument",
    )
    parser.add_argument("-o", "--output-file", required=True, help="Output file name")
    args = parser.parse_args()
    template_dir, template_file_name = os.path.split(args.template_file)
    output_file_name = args.output_file

    output = render(template_dir, template_file_name)
    with open(output_file_name, "w", encoding="utf-8") as file:
        file.write(output)
    print(f"Written JSON file: {output_file_name}")


if __name__ == "__main__":
    main()
