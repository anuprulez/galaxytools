import math
import sys

from ij import IJ, ImagePlus, Prefs
from ij.gui import Plot
from ij.io import FileSaver
from java.awt import Color, Font
from java.io import File
from jarray import array
from sc.fiji.analyzeSkeleton import AnalyzeSkeleton_


def percentile(sorted_values, fraction):
    if not sorted_values:
        return 0.0
    position = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def write_summary(path, diameters, diameters_pixels, unit, scale, width, height, pore_areas, porosity,
                  fiber_area, sensitive_length, voronoi_length, intersection_count, three_points,
                  four_points, characteristic_length, characteristic_sd, characteristic_max,
                  super_pixel_diameter, corrected_count, saturated_samples):
    count = len(diameters)
    mean = sum(diameters) / count
    variance = sum([(value - mean) ** 2 for value in diameters]) / count
    sorted_values = sorted(diameters)
    outf = open(path, "wb")
    outf.write("Metric\tValue\tUnit\n")
    outf.write("Centerline samples\t%d\tcount\n" % count)
    outf.write("Mean diameter\t%.6f\t%s\n" % (mean, unit))
    outf.write("Standard deviation\t%.6f\t%s\n" % (math.sqrt(variance), unit))
    outf.write("Minimum diameter\t%.6f\t%s\n" % (sorted_values[0], unit))
    outf.write("Median diameter\t%.6f\t%s\n" % (percentile(sorted_values, 0.5), unit))
    outf.write("Maximum diameter\t%.6f\t%s\n" % (sorted_values[-1], unit))
    frequencies = {}
    for value in diameters:
        frequencies[value] = frequencies.get(value, 0) + 1
    mode = max(frequencies.keys(), key=lambda value: frequencies[value])
    outf.write("Mode diameter\t%.6f\t%s\n" % (mode, unit))
    if variance > 0:
        skewness = sum([(value - mean) ** 3 for value in diameters]) / count / math.pow(variance, 1.5)
        kurtosis = sum([(value - mean) ** 4 for value in diameters]) / count / (variance * variance) - 3.0
    else:
        skewness = 0.0
        kurtosis = 0.0
    outf.write("Diameter skewness\t%.6f\tdimensionless\n" % skewness)
    outf.write("Diameter excess kurtosis\t%.6f\tdimensionless\n" % kurtosis)
    outf.write("Diameter raw integrated density\t%.6f\t%s\n" % (sum(diameters) / 2.0, unit))
    outf.write("Pixel scale\t%.6f\t%s/pixel\n" % (scale, unit))
    outf.write("Fiber area\t%.6f\t%s^2\n" % (fiber_area * scale * scale, unit))
    outf.write("Sensitive centerline length\t%.6f\t%s\n" % (sensitive_length * scale, unit))
    outf.write("Voronoi centerline length\t%.6f\t%s\n" % (voronoi_length * scale, unit))
    outf.write("Super-pixel diameter\t%.6f\t%s\n" % (super_pixel_diameter * scale, unit))
    outf.write("Intersections\t%d\tcount\n" % intersection_count)
    outf.write("Three-point intersections\t%d\tcount\n" % three_points)
    outf.write("Four-point intersections\t%d\tcount\n" % four_points)
    outf.write("Intersection density\t%.6f\tper 10000 pixels\n" %
               (intersection_count * 10000.0 / (width * height)))
    outf.write("Characteristic fiber length\t%.6f\t%s\n" % (characteristic_length * scale, unit))
    outf.write("Characteristic fiber length SD\t%.6f\t%s\n" % (characteristic_sd * scale, unit))
    outf.write("Maximum branch length\t%.6f\t%s\n" % (characteristic_max * scale, unit))
    outf.write("Intersection-corrected samples\t%d\tcount\n" % corrected_count)
    upper_limit = min(512.0, 0.1 * min(width, height))
    suitable = [value for value in diameters_pixels if value >= 10.0 and value <= upper_limit]
    suitable_fraction = 100.0 * len(suitable) / len(diameters_pixels)
    status = "PASS" if suitable_fraction == 100.0 and saturated_samples == 0 else "REVIEW"
    outf.write("DiameterJ suitability\t%s\tstatus\n" % status)
    outf.write("Validated diameter interval\t10.000000-%.6f\tpixel\n" % upper_limit)
    outf.write("Samples inside validated interval\t%.3f\tpercent\n" % suitable_fraction)
    outf.write("Saturated EDT samples\t%d\tcount\n" % saturated_samples)
    outf.write("Mesh-hole count\t%d\tcount\n" % len(pore_areas))
    outf.write("Percent porosity\t%.6f\tpercent\n" % porosity)
    if pore_areas:
        pore_mean = sum(pore_areas) / len(pore_areas)
        pore_variance = sum([(value - pore_mean) ** 2 for value in pore_areas]) / len(pore_areas)
        outf.write("Total enclosed mesh-hole area\t%.6f\t%s^2\n" % (sum(pore_areas), unit))
        outf.write("Mean mesh-hole area\t%.6f\t%s^2\n" % (pore_mean, unit))
        outf.write("Mesh-hole area SD\t%.6f\t%s^2\n" % (math.sqrt(pore_variance), unit))
        outf.write("Minimum mesh-hole area\t%.6f\t%s^2\n" % (min(pore_areas), unit))
        outf.write("Maximum mesh-hole area\t%.6f\t%s^2\n" % (max(pore_areas), unit))
    outf.close()


def make_histogram(diameters, bin_width):
    bins = {}
    for diameter in diameters:
        index = int(math.floor(diameter / bin_width))
        bins[index] = bins.get(index, 0) + 1
    return bins


def write_histogram(path, bins, bin_width, unit):
    outf = open(path, "wb")
    outf.write("Diameter_bin_start\tDiameter_bin_end\tCount\tUnit\n")
    for index in sorted(bins.keys()):
        lower = index * bin_width
        upper = lower + bin_width
        outf.write("%.6f\t%.6f\t%d\t%s\n" % (lower, upper, bins[index], unit))
    outf.close()


def analyze_mesh_holes(binary_pixels, width, height, scale, minimum_area):
    visited = set()
    areas = []
    background_count = 0
    size = width * height
    for start in range(size):
        if (binary_pixels[start] & 0xff) != 0:
            continue
        background_count += 1
        if start in visited:
            continue
        visited.add(start)
        stack = [start]
        area = 0
        touches_edge = False
        while stack:
            index = stack.pop()
            area += 1
            x = index % width
            y = index // width
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                touches_edge = True
            for yy in range(max(0, y - 1), min(height, y + 2)):
                for xx in range(max(0, x - 1), min(width, x + 2)):
                    neighbor = yy * width + xx
                    if neighbor not in visited and (binary_pixels[neighbor] & 0xff) == 0:
                        visited.add(neighbor)
                        stack.append(neighbor)
        if not touches_edge and area >= minimum_area:
            areas.append(area * scale * scale)
    return areas, 100.0 * background_count / size


def write_mesh_holes(path, areas, unit):
    outf = open(path, "wb")
    outf.write("Mesh_hole\tArea\tUnit\n")
    for index, area in enumerate(sorted(areas, reverse=True)):
        outf.write("%d\t%.6f\t%s^2\n" % (index + 1, area, unit))
    outf.close()


def write_mesh_hole_plot(path, areas, bin_width, unit):
    bins = make_histogram(areas, bin_width) if areas else {0: 0}
    indices = sorted(bins.keys())
    x_values = [(indices[0] - 0.5) * bin_width]
    y_values = [0.0]
    for index in indices:
        x_values.append((index + 0.5) * bin_width)
        y_values.append(float(bins[index]))
    x_values.append((indices[-1] + 1.5) * bin_width)
    y_values.append(0.0)
    plot = Plot("DiameterJ mesh-hole histogram", "Mesh-hole area (%s^2)" % unit, "Frequency")
    plot.setSize(700, 500)
    plot.setColor(Color.black)
    plot.add("filled", array(x_values, "d"), array(y_values, "d"))
    plot.setFont("f", Font("SansSerif", Font.PLAIN, 16))
    plot.setFont("x", Font("SansSerif", Font.PLAIN, 20))
    plot.setFont("y", Font("SansSerif", Font.PLAIN, 20))
    image = ImagePlus("DiameterJ mesh-hole histogram", plot.getProcessor())
    saved = FileSaver(image).saveAsPng(path)
    if not saved or not File(path).isFile() or File(path).length() == 0:
        raise RuntimeError("ImageJ did not create the mesh-hole histogram PNG")


def neighbor_count(pixels, width, height, index):
    x = index % width
    y = index // width
    count = 0
    for yy in range(max(0, y - 1), min(height, y + 2)):
        for xx in range(max(0, x - 1), min(width, x + 2)):
            other = yy * width + xx
            if other != index and (pixels[other] & 0xff) != 0:
                count += 1
    return count


def centerline_pixel_count(pixels):
    return sum([1 for value in pixels if (value & 0xff) != 0])


def analyze_skeleton_metrics(skeleton):
    analyzer = AnalyzeSkeleton_()
    working = skeleton.duplicate()
    analyzer.setup("", working)
    result = analyzer.run(analyzer.SHORTEST_BRANCH, False, False, working, True, True)
    junctions = sum([int(value) for value in result.getJunctions()])
    quadruples = sum([int(value) for value in result.getQuadruples()])
    three_points = max(0, junctions - quadruples)
    branch_lengths = []
    for graph in result.getGraph():
        for edge in graph.getEdges():
            branch_lengths.append(float(edge.getLength()))
    if branch_lengths:
        mean = sum(branch_lengths) / len(branch_lengths)
        variance = sum([(value - mean) ** 2 for value in branch_lengths]) / len(branch_lengths)
        return junctions, three_points, quadruples, mean, math.sqrt(variance), max(branch_lengths)
    return junctions, three_points, quadruples, 0.0, 0.0, 0.0


def find_intersections(pixels, distance_pixels, distance_is_byte, width, height):
    candidates = set()
    for index in range(width * height):
        if (pixels[index] & 0xff) != 0 and neighbor_count(pixels, width, height, index) >= 3:
            candidates.add(index)
    intersections = []
    while candidates:
        start = candidates.pop()
        component = [start]
        stack = [start]
        while stack:
            index = stack.pop()
            x = index % width
            y = index // width
            for yy in range(max(0, y - 1), min(height, y + 2)):
                for xx in range(max(0, x - 1), min(width, x + 2)):
                    other = yy * width + xx
                    if other in candidates:
                        candidates.remove(other)
                        component.append(other)
                        stack.append(other)
        component_set = set(component)
        arms = set()
        for index in component:
            x = index % width
            y = index // width
            for yy in range(max(0, y - 1), min(height, y + 2)):
                for xx in range(max(0, x - 1), min(width, x + 2)):
                    other = yy * width + xx
                    if other not in component_set and (pixels[other] & 0xff) != 0:
                        arms.add(other)
        cx = sum([index % width for index in component]) / float(len(component))
        cy = sum([index // width for index in component]) / float(len(component))
        center = component[len(component) // 2]
        radius = distance_pixels[center]
        if distance_is_byte:
            radius = radius & 0xff
        arm_count = min(4, max(3, len(arms)))
        intersections.append({"pixels": component, "x": cx, "y": cy,
                              "arms": arm_count, "radius": float(radius)})
    return intersections


def write_intersections(path, intersections, scale, unit):
    outf = open(path, "wb")
    outf.write("Intersection\tX\tY\tEstimated_branches\tLocal_radius\tUnit\n")
    for index, item in enumerate(intersections):
        outf.write("%d\t%.3f\t%.3f\t%d\t%.6f\t%s\n" %
                   (index + 1, item["x"], item["y"], item["arms"], item["radius"] * scale, unit))
    outf.close()


def correct_intersections(skeleton_pixels, distance_pixels, distance_is_byte, width, height, intersections):
    junctions = []
    for item in intersections:
        junctions.append((item["x"], item["y"], max(1.0, item["radius"] / math.sqrt(2.0))))
    corrected = []
    excluded = set()
    for index in range(width * height):
        if (skeleton_pixels[index] & 0xff) == 0:
            continue
        distance = distance_pixels[index]
        if distance_is_byte:
            distance = distance & 0xff
        if distance <= 0:
            continue
        x = index % width
        y = index // width
        reject = False
        for jx, jy, radius in junctions:
            if (x - jx) ** 2 + (y - jy) ** 2 <= radius ** 2:
                reject = True
                break
        if reject:
            excluded.add(index)
        else:
            corrected.append(2.0 * float(distance))
    return corrected, excluded


def write_diagnostic(path, binary, skeleton_pixels, excluded, intersections, width, height):
    diagnostic = binary.duplicate()
    IJ.run(diagnostic, "RGB Color", "")
    processor = diagnostic.getProcessor()
    processor.setColor(Color.red)
    for index in range(width * height):
        if (skeleton_pixels[index] & 0xff) != 0:
            processor.drawPixel(index % width, index // width)
    processor.setColor(Color.yellow)
    for index in excluded:
        processor.drawPixel(index % width, index // width)
    processor.setColor(Color.cyan)
    for item in intersections:
        processor.drawOval(int(item["x"]) - 2, int(item["y"]) - 2, 5, 5)
    save_tiff(diagnostic, path, "intersection diagnostic")


def analyze_orientation(skeleton_pixels, width, height, radius):
    orientations = []
    for index in range(width * height):
        if (skeleton_pixels[index] & 0xff) == 0:
            continue
        center_x = index % width
        center_y = index // width
        points = []
        for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
            row = y * width
            for x in range(max(0, center_x - radius), min(width, center_x + radius + 1)):
                if (skeleton_pixels[row + x] & 0xff) != 0:
                    points.append((float(x), float(y)))
        if len(points) < 3:
            continue
        mean_x = sum([point[0] for point in points]) / len(points)
        mean_y = sum([point[1] for point in points]) / len(points)
        xx = sum([(point[0] - mean_x) ** 2 for point in points])
        yy = sum([(point[1] - mean_y) ** 2 for point in points])
        xy = sum([(point[0] - mean_x) * (point[1] - mean_y) for point in points])
        angle = math.degrees(0.5 * math.atan2(2.0 * xy, xx - yy)) % 180.0
        orientations.append(angle)
    return orientations


def write_orientation(path, bins, bin_width):
    outf = open(path, "wb")
    outf.write("Angle_bin_start\tAngle_bin_end\tCount\tUnit\n")
    for index in range(int(math.ceil(180.0 / bin_width))):
        outf.write("%.6f\t%.6f\t%d\tdegree\n" %
                   (index * bin_width, min(180.0, (index + 1) * bin_width), bins.get(index, 0)))
    outf.close()


def write_orientation_plot(path, bins, bin_width):
    count = int(math.ceil(180.0 / bin_width))
    x_values = [0.0]
    y_values = [0.0]
    for index in range(count):
        x_values.append((index + 0.5) * bin_width)
        y_values.append(float(bins.get(index, 0)))
    x_values.append(180.0)
    y_values.append(0.0)
    plot = Plot("DiameterJ fiber orientation", "Fiber orientation (degrees)", "Frequency")
    plot.setSize(700, 500)
    plot.setLimits(0.0, 180.0, 0.0, max(y_values) * 1.05 if max(y_values) else 1.0)
    plot.setColor(Color.black)
    plot.add("filled", array(x_values, "d"), array(y_values, "d"))
    plot.setFont("f", Font("SansSerif", Font.PLAIN, 16))
    plot.setFont("x", Font("SansSerif", Font.PLAIN, 20))
    plot.setFont("y", Font("SansSerif", Font.PLAIN, 20))
    image = ImagePlus("DiameterJ fiber orientation", plot.getProcessor())
    saved = FileSaver(image).saveAsPng(path)
    if not saved or not File(path).isFile() or File(path).length() == 0:
        raise RuntimeError("ImageJ did not create the fiber orientation PNG")


def write_histogram_plot(path, bins, bin_width, unit):
    indices = sorted(bins.keys())
    # Add zero-height endpoints so the filled curve closes at the baseline.
    x_values = [(indices[0] - 0.5) * bin_width]
    y_values = [0.0]
    for index in indices:
        x_values.append((index + 0.5) * bin_width)
        y_values.append(float(bins[index]))
    x_values.append((indices[-1] + 1.5) * bin_width)
    y_values.append(0.0)

    if unit == "pixel":
        x_label = "Fiber diameter (pixels)"
    else:
        x_label = "Fiber diameter (%s)" % unit
    plot = Plot("DiameterJ diameter histogram", x_label, "Frequency")
    plot.setSize(700, 500)
    plot.setColor(Color.black)
    plot.add("filled", array(x_values, "d"), array(y_values, "d"))
    plot.setFont("f", Font("SansSerif", Font.PLAIN, 16))
    plot.setFont("x", Font("SansSerif", Font.PLAIN, 20))
    plot.setFont("y", Font("SansSerif", Font.PLAIN, 20))

    mode_index = max(indices, key=lambda index: bins[index])
    mode = (mode_index + 0.5) * bin_width
    x_span = x_values[-1] - x_values[0]
    label_x = (mode - x_values[0]) / x_span if x_span else 0.5
    label_x = min(max(label_x, 0.05), 0.85)
    plot.addLabel(label_x, 0.05, "%.2f" % mode)
    # getImagePlus() can remain an undrawn, blank canvas in headless Fiji.
    # getProcessor() forces Plot to rasterize axes, labels, and data first.
    plot_processor = plot.getProcessor()
    plot_image = ImagePlus("DiameterJ diameter histogram", plot_processor)
    saved = FileSaver(plot_image).saveAsPng(path)
    if not saved or not File(path).isFile() or File(path).length() == 0:
        raise RuntimeError("ImageJ did not create the diameter histogram PNG")


def save_tiff(image, path, description):
    saved = FileSaver(image).saveAsTiff(path)
    if not saved or not File(path).isFile() or File(path).length() == 0:
        raise RuntimeError("ImageJ did not create the %s TIFF" % description)


# Fiji's Jython is Python 2, so positional arguments are used instead of argparse.
input_path = sys.argv[-21]
foreground = sys.argv[-20]
scale = float(sys.argv[-19])
unit = sys.argv[-18]
bin_width_pixels = float(sys.argv[-17])
orientation_bin_width = float(sys.argv[-16])
orientation_radius = int(sys.argv[-15])
mesh_bin_width_pixels = float(sys.argv[-14])
minimum_mesh_area = int(sys.argv[-13])
summary_path = sys.argv[-12]
histogram_path = sys.argv[-11]
histogram_plot_path = sys.argv[-10]
mesh_holes_path = sys.argv[-9]
mesh_plot_path = sys.argv[-8]
orientation_path = sys.argv[-7]
orientation_plot_path = sys.argv[-6]
intersections_path = sys.argv[-5]
diagnostic_path = sys.argv[-4]
distance_map_path = sys.argv[-3]
skeleton_path = sys.argv[-2]
voronoi_path = sys.argv[-1]

image = IJ.openImage(input_path)
if image is None:
    raise RuntimeError("ImageJ could not open the input image")

binary = image.duplicate()
IJ.run(binary, "8-bit", "")
processor = binary.getProcessor()
input_is_binary = processor.isBinary()
if not input_is_binary:
    threshold_method = "Default dark" if foreground == "white" else "Default"
    IJ.setAutoThreshold(binary, threshold_method)
    Prefs.blackBackground = True
    IJ.run(binary, "Convert to Mask", "")

# DiameterJ expects white fibers on a black background.
if input_is_binary and foreground == "black":
    binary.getProcessor().invert()

# Use the ImageJ command rather than EDM.makeFloatEDM/makeFloat.  The public
# EDM Java API differs between the Fiji releases packaged for Galaxy, while
# this command is stable and is also the command used by DiameterJ itself.
distance_image = binary.duplicate()
IJ.run(distance_image, "Distance Map", "")
distance_processor = distance_image.getProcessor()
save_tiff(distance_image, distance_map_path, "distance map")

skeleton = binary.duplicate()
IJ.run(skeleton, "Skeletonize", "")
save_tiff(skeleton, skeleton_path, "fiber centerline")

# DiameterJ's insensitive centerline is based on a Voronoi transform.
voronoi = binary.duplicate()
voronoi.getProcessor().invert()
IJ.run(voronoi, "Voronoi", "")
if not voronoi.getProcessor().isBinary():
    IJ.run(voronoi, "Make Binary", "")
# Make Binary follows a global ImageJ background preference. Voronoi lines
# are the minority class, so normalize them to white deterministically.
voronoi_raw_pixels = voronoi.getProcessor().getPixels()
if centerline_pixel_count(voronoi_raw_pixels) > voronoi.getWidth() * voronoi.getHeight() / 2:
    voronoi.getProcessor().invert()
IJ.run(voronoi, "Skeletonize", "")
save_tiff(voronoi, voronoi_path, "Voronoi centerline")

skeleton_pixels = skeleton.getProcessor().getPixels()
binary_pixels = binary.getProcessor().getPixels()
distance_pixels = distance_processor.getPixels()
diameters = []
diameters_pixels = []
saturated_samples = 0
for index in range(len(skeleton_pixels)):
    distance = distance_pixels[index]
    # Jython exposes ImageJ byte pixels as signed Java bytes.
    if distance_image.getBitDepth() == 8:
        distance = distance & 0xff
    if (skeleton_pixels[index] & 0xff) != 0 and distance > 0:
        if distance_image.getBitDepth() == 8 and distance == 255:
            saturated_samples += 1
        diameters_pixels.append(2.0 * float(distance))
        diameters.append(2.0 * float(distance) * scale)

if not diameters:
    raise RuntimeError("No fiber centerline pixels were found; check the foreground setting and segmentation")

width = binary.getWidth()
height = binary.getHeight()
distance_is_byte = distance_image.getBitDepth() == 8
intersections = find_intersections(skeleton_pixels, distance_pixels, distance_is_byte, width, height)
corrected_pixels, excluded = correct_intersections(
    skeleton_pixels, distance_pixels, distance_is_byte, width, height, intersections
)
if corrected_pixels:
    diameters_pixels = corrected_pixels
    diameters = [value * scale for value in corrected_pixels]
write_intersections(intersections_path, intersections, scale, unit)
write_diagnostic(diagnostic_path, binary, skeleton_pixels, excluded, intersections, width, height)

fiber_area = sum([1 for value in binary_pixels if (value & 0xff) != 0])
sensitive_length = centerline_pixel_count(skeleton_pixels)
voronoi_pixels = voronoi.getProcessor().getPixels()
voronoi_length = centerline_pixel_count(voronoi_pixels)
(intersection_count, three_points, four_points, characteristic_length,
 characteristic_sd, characteristic_max) = analyze_skeleton_metrics(skeleton)
super_pixel_diameter = fiber_area / sensitive_length if sensitive_length > 0 else 0.0
for iteration in range(100):
    medial_corrected = sensitive_length - three_points * 0.5 * super_pixel_diameter - four_points * super_pixel_diameter
    if medial_corrected <= 0:
        break
    updated = fiber_area / medial_corrected
    if abs(updated - super_pixel_diameter) < 0.001:
        super_pixel_diameter = updated
        break
    super_pixel_diameter = updated
medial_corrected = max(1.0, sensitive_length - three_points * 0.5 * super_pixel_diameter -
                       four_points * super_pixel_diameter)
voronoi_corrected = max(1.0, voronoi_length - three_points * 0.5 * super_pixel_diameter -
                        four_points * super_pixel_diameter)
medial_diameter = fiber_area / medial_corrected
voronoi_diameter = fiber_area / voronoi_corrected
super_pixel_diameter = (medial_diameter + voronoi_diameter) / 2.0

pore_areas, porosity = analyze_mesh_holes(binary_pixels, width, height, scale, minimum_mesh_area)
write_mesh_holes(mesh_holes_path, pore_areas, unit)
write_mesh_hole_plot(mesh_plot_path, pore_areas, mesh_bin_width_pixels * scale * scale, unit)

orientations = analyze_orientation(skeleton_pixels, width, height, orientation_radius)
orientation_bins = make_histogram(orientations, orientation_bin_width)
write_orientation(orientation_path, orientation_bins, orientation_bin_width)
write_orientation_plot(orientation_plot_path, orientation_bins, orientation_bin_width)

write_summary(summary_path, diameters, diameters_pixels, unit, scale, width, height, pore_areas, porosity,
              fiber_area, sensitive_length, voronoi_length, intersection_count, three_points, four_points,
              characteristic_length, characteristic_sd, characteristic_max, super_pixel_diameter,
              len(diameters_pixels), saturated_samples)
bin_width = bin_width_pixels * scale
bins = make_histogram(diameters, bin_width)
write_histogram(histogram_path, bins, bin_width, unit)
write_histogram_plot(histogram_plot_path, bins, bin_width, unit)
