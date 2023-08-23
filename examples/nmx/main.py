import argparse
import json
import os
from pathlib import PurePosixPath
from operator import itemgetter

import jinja2

try:
    from nx_detector import BoxNXDetector
    from nx_sample import NXSample
except ModuleNotFoundError:
    from examples.nmx.nx_detector import BoxNXDetector
    from examples.nmx.nx_sample import NXSample


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
    print("Creating sample and detectors...")

    detector_panel_0 = BoxNXDetector(
        PurePosixPath("/entry/instrument"),
        "detector_panel_0",
        "nmx",
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
    # # Example of manual positioning instead of using NXlog:
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
    # Detector positioning from NXlog:
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
        PurePosixPath("/entry/instrument"),
        "detector_panel_1",
        "nmx",
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
    # # Example of manual positioning instead of using NXlog:
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
    # Detector positioning from NXlog:
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
        PurePosixPath("/entry/instrument"),
        "detector_panel_2",
        "nmx",
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
    # # Example of manual positioning instead of using NXlog:
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
    # Detector positioning from NXlog:
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

    # # sample at zero
    # sample = NXSample(name="sample", sample_name="sample name", instrument_name="nmx")
    # (
    #     sample
    #     .translate("y_offset_sample_to_samplebase", y=-1.082)
    #     .rotate("axis1", y=0)
    #     .translate("x_offset_axis1_to_axis2", x=0.148)
    #     .translate("y_offset_axis2_to_axis2", y=0.232)
    #     .rotate("axis2", x=-0)
    #     .translate("y_offset_axis2_to_axis3", y=0.425)
    #     .rotate("axis3", x=-0)
    #     .translate("x_offset_axis3_to_axis4", x=-0.1105)
    #     .translate("y_offset_axis3_to_axis4", y=0.2)
    #     .rotate("axis4", y=0)
    #     .translate("y_offset_axis4_to_axis5", y=0.125)
    #     .rotate("axis5", x=0)
    #     .translate("y_offset_axis5_to_axis6", y=0.1)
    #     .rotate("axis6", y=90)
    # )
    # # sample with example rotations:
    # sample = NXSample(name="sample", sample_name="sample name", instrument_name="nmx")
    # (
    #     sample
    #     .translate("y_offset_sample_to_samplebase", y=-1.082)
    #     .rotate("axis1", y=30)
    #     .translate("x_offset_axis1_to_axis2", x=0.148)
    #     .translate("y_offset_axis2_to_axis2", y=0.232)
    #     .rotate("axis2", x=-30)
    #     .translate("y_offset_axis2_to_axis3", y=0.425)
    #     .rotate("axis3", x=-30)
    #     .translate("x_offset_axis3_to_axis4", x=-0.1105)
    #     .translate("y_offset_axis3_to_axis4", y=0.2)
    #     .rotate("axis4", y=45)
    #     .translate("y_offset_axis4_to_axis5", y=0.1)
    #     .rotate("axis5", x=45)
    #     .translate("y_offset_axis5_to_axis6", y=0.1)
    #     .rotate("axis6", y=30)
    # )
    # Sample positioning from NXlog:
    sample = NXSample(parent=PurePosixPath("/entry"), name="sample", sample_name="sample name", instrument_name="nmx")
    (
        sample
        .translate("y_offset_sample_to_samplebase", y=-1.082)
        .rotate_from_nxlog("axis1", [0,1,0], "f144", "nmx_robot", "NMX:SampleRobot:axis1")
        .translate("x_offset_axis1_to_axis2", x=0.148)
        .translate("y_offset_axis2_to_axis2", y=0.232)
        .rotate_from_nxlog("axis2", [1,0,0], "f144", "nmx_robot", "NMX:SampleRobot:axis2")
        .translate("y_offset_axis2_to_axis3", y=0.425)
        .rotate_from_nxlog("axis3", [1,0,0], "f144", "nmx_robot", "NMX:SampleRobot:axis3")
        .translate("x_offset_axis3_to_axis4", x=-0.1105)
        .translate("y_offset_axis3_to_axis4", y=0.2)
        .rotate_from_nxlog("axis4", [0,1,0], "f144", "nmx_robot", "NMX:SampleRobot:axis4")
        .translate("y_offset_axis4_to_axis5", y=0.125)
        .rotate_from_nxlog("axis5", [1,0,0], "f144", "nmx_robot", "NMX:SampleRobot:axis5")
        .translate("y_offset_axis5_to_axis6", y=0.1)
        .rotate_from_nxlog("axis6", [0,1,0], "f144", "nmx_robot", "NMX:SampleRobot:axis6")
    )


    # rendering

    print(f"Rendering detectors...")
    context = {
        # "j2_instrument_name": instrument_name,
        "j2_instrument_detector_panel_0": detector_panel_0.to_json(),
        "j2_instrument_detector_panel_1": detector_panel_1.to_json(),
        "j2_instrument_detector_panel_2": detector_panel_2.to_json(),
    }

    print(f"Rendering sample...")
    context.update({
        "j2_nxsample": sample.to_json(),
    })

    print(f"Rendering template...")
    output = render_template(template_dir, template_file_name, **context)
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
    json.loads(output)  # raises json.JSONDecodeError if invalid JSON
    with open(output_file_name + ".sorted.json", "w", encoding="utf-8") as file_sorted:
        json.dump(custom_sort(output, CUSTOM_SORT_ORDER), file_sorted, indent=2, sort_keys=False)
    print(f"Written JSON file: {output_file_name}")


def custom_sort(obj, key_order):
    if isinstance(obj, dict):
        sorted_keys = sorted(obj.keys(), key=lambda k: key_order.get(k, key_order.get('__default__', float('inf'))))
        return {k: custom_sort(obj[k], key_order) for k in sorted_keys}
    else:
        return obj


CUSTOM_SORT_ORDER = {
    'name': 10,
    'module': 30,
    'values': 31,
    'config': 40,
    'type': 40,
    'dtype': 40,
    '__default__': 80,  # Default order for unknown keys
    'attributes': 90,
    'children': 99,
}


if __name__ == "__main__":
    main()
