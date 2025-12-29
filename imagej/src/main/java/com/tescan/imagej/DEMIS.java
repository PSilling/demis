/**
 * Deep Electron Microscopy Image Stitching (DEMIS) runner plugin for ImageJ.
 * 
 * @author Petr Šilling
 * @year 2025
 */

package com.tescan.imagej;

import net.imagej.ImageJ;
import net.imglib2.type.numeric.RealType;

import org.scijava.ItemVisibility;
import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * DEMIS operator plugin for ImageJ.
 * 
 * The plugin works by running DEMIS CLI in the background and displaying the
 * result.
 */
@Plugin(type = Command.class, menuPath = "Plugins>Stitching>DEMIS")
public class DEMIS<T extends RealType<T>> implements Command {
    @Parameter(label = "DEMIS Project Paths", visibility = ItemVisibility.MESSAGE)
    private String demisPathsLabel = "";

    @Parameter(
        label = "DEMIS directory",
        description = "Path to the top-level DEMIS installation directory.",
        style = "directory"
    )
    private File demisDir;

    @Parameter(
        label = "Python executable",
        description = "Path to the Python executable with DEMIS dependencies installed.",
        style = "file"
    )
    private File pythonExe;

    @Parameter(
        label = "LoFTR checkpoint path",
        description = "Path to the LoFTR model weights file. Can be omitted if not using LoFTR matching.",
        required = false,
        style = "file"
    )
    private File loftrCheckpointPath = null;

    @Parameter(
        label = "YAML config file",
        description = "Path to a DEMIS configuration file. If omitted, the parameters below can be used to override "
            + "default DEMIS settings.",
        required = false,
        style = "file"
    )
    private File yamlConfig = null;

    @Parameter(label = " ", visibility = ItemVisibility.MESSAGE)
    private String datasetSettingsLabelMargin = "";

    @Parameter(label = "Dataset Settings", visibility = ItemVisibility.MESSAGE)
    private String datasetSettingsLabel = "";

    @Parameter(
        label = "Source dataset directory",
        description = "Path to the directory containing the image tiles to be stitched.",
        style = "directory"
    )
    private File inputDir;

    @Parameter(
        label = "Output directory",
        description = "Path to the directory where the stitched output will be saved.",
        style = "directory"
    )
    private File outputDir;

    @Parameter(
        label = "Open output after stitching",
        description = "Whether to open the stitched images in ImageJ after stitching. Warning: this may consume a lot "
            + "of memory for large datasets."
    )
    private Boolean openAfterStitching = true;

    @Parameter(
        label = "Grid rows",
        description = "Number of rows in the image tile grid.",
        min = "1"
    )
    private Integer gridRows = 2;

    @Parameter(
        label = "Grid columns",
        description = "Number of columns in the image tile grid.",
        min = "1"
    )
    private Integer gridCols = 2;

    @Parameter(label = " ", visibility = ItemVisibility.MESSAGE)
    private String stitchingParametersLabelMargin = "";

    @Parameter(label = "Stitching Parameters (Only Used if Config is Empty)", visibility = ItemVisibility.MESSAGE)
    private String stitchingParametersLabel = "";

    @Parameter(
        label = "Image tile overlap",
        description = "Approximate overlap between adjacent tiles as a fraction of tile size (e.g., 0.3 for 30% overlap).",
        min = "0.01",
        max = "0.99",
        stepSize = "0.01"
    )
    private Double tileOverlap = 0.3;

    @Parameter(
        label = "Matching resolution ratio",
        description = "Ratio of the resolution used for matching features between adjacent tiles.",
        min = "0.1",
        max = "1.0",
        stepSize = "0.1"
    )
    private Double resolutionRatio = 0.5;

    @Parameter(
        label = "Matching method",
        description = "Method for matching features between adjacent tiles.",
        choices = { "loftr", "sift", "orb" }
    )
    private String matchingMethod = "loftr";

    @Parameter(
        label = "Transform type",
        description = "Type of geometric transformation to apply during stitching. Warning: not all methods are "
            + "compatible with all matching methods.",
        choices = { "translation", "euclidean", "similarity", "affine", "projective" }
    )
    private String transformType = "euclidean";

    @Parameter(
        label = "Stitched image construction method",
        description = "Method for constructing the final stitched image from individual tiles.",
        choices = { "optimised", "mst", "slam" }
    )
    private String constructionMethod = "optimised";

    @Parameter(
        label = "Compositing method",
        description = "Method for blending overlapping regions of tiles in the final stitched image.",
        choices = { "overwrite", "average", "adaptive" }
    )
    private String compositingMethod = "overwrite";

    @Parameter(
        label = "Optical flow refinement",
        description = "Whether to apply optical flow refinement to improve alignment in overlapping regions."
    )
    private Boolean opticalFlowRefinement = true;

    @Parameter(
        label = "Optical flow refinement type",
        description = "Type of optical flow refinement to use.",
        choices = { "grid", "overlap" }
    )
    private String opticalFlowRefinementType = "grid";

    @Parameter(
        label = "Normalise intensity",
        description = "Whether to normalise the intensity of the stitched image. Note that intensity normalisation is "
            + "done on individual tiles only and does not guarantee consistency across the entire stitched image."
    )
    private Boolean normaliseIntensity = true;

    @Parameter(
        label = "Colored output",
        description = "Whether to give randomized colors to each stitched tile in the final output image. Useful for "
            + "visualizing tile boundaries."
    )
    private Boolean coloredOutput = false;

    /** ImageJ */
    @Parameter
    private ImageJ ij;

    /** ImageJ logging service. */
    @Parameter
    private LogService log;

    /**
     * Entry point of the DEMIS CLI plugin.
     */
    @Override
    public void run() {
        List<String> cmd = buildDEMISCmd();
        runSticher(cmd);
    }

    /**
     * Build the command for DEMIS CLI using the given parameters.
     * 
     * @return The command as a list of strings.
     */
    private List<String> buildDEMISCmd() {
        List<String> cmd = new ArrayList<>();
        Path pathToStitchScript = Paths.get(demisDir.getAbsolutePath(), "scripts/stitch.py");

        // Base command: python stitch.py --plugin-mode
        cmd.add(pythonExe.getAbsolutePath());
        cmd.add(pathToStitchScript.toAbsolutePath().toString());
        cmd.add("--plugin-mode");

        if (yamlConfig != null && yamlConfig.exists()) {
            // If using YAML config, pass the config only.
            cmd.add("--config");
            cmd.add(yamlConfig.getAbsolutePath());
        } else {
            // No config, build the full command using all optional stitching parameters.
            cmd.add("DATASET.TILE_OVERLAP");
            cmd.add(tileOverlap.toString());

            cmd.add("STITCHER.RESOLUTION_RATIO");
            cmd.add(resolutionRatio.toString());

            cmd.add("STITCHER.MATCHING_METHOD");
            cmd.add(matchingMethod);

            cmd.add("STITCHER.TRANSFORM_TYPE");
            cmd.add(transformType);

            cmd.add("STITCHER.CONSTRUCTION_METHOD");
            cmd.add(constructionMethod);

            cmd.add("STITCHER.COMPOSITING_METHOD");
            cmd.add(compositingMethod);

            cmd.add("STITCHER.OPTICAL_FLOW_REFINEMENT");
            cmd.add(opticalFlowRefinement ? "True" : "False");

            cmd.add("STITCHER.OPTICAL_FLOW_REFINEMENT_TYPE");
            cmd.add(opticalFlowRefinementType);

            cmd.add("STITCHER.COLORED_OUTPUT");
            cmd.add(coloredOutput ? "True" : "False");

            cmd.add("STITCHER.NORMALISE_INTENSITY");
            cmd.add(normaliseIntensity ? "True" : "False");

            cmd.add("LOFTR.CHECKPOINT_PATH");
            cmd.add(loftrCheckpointPath.getAbsolutePath());
        }
        
        // Mandatory parameter overrides.
        cmd.add("DATASET.PATH");
        cmd.add(inputDir.getAbsolutePath());

        cmd.add("STITCHER.OUTPUT_PATH");
        cmd.add(outputDir.getAbsolutePath());

        cmd.add("DATASET.ROWS");
        cmd.add(gridRows.toString());

        cmd.add("DATASET.COLS");
        cmd.add(gridCols.toString());

        return cmd;
    }

    /**
     * Runs the DEMIS stitcher in a subprocess, logging all output.
     * 
     * @param cmd Command that runs DEMIS.
     * @return Subprocess exit code.
     */
    private int runSticher(List<String> cmd) {
        int exitCode = -1;

        try {
            ProcessBuilder pb = new ProcessBuilder(cmd);
            Map<String, String> pbEnv = pb.environment();
            pbEnv.put("PYTHONPATH", demisDir.getAbsolutePath());
            Process process = pb.start();

            // Stdout logging.
            try (BufferedReader outReader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = outReader.readLine()) != null) {
                    log.info(line);
                }
            }

            // Stderr logging.
            try (BufferedReader errReader = new BufferedReader(new InputStreamReader(process.getErrorStream()))) {
                String line;
                while ((line = errReader.readLine()) != null) {
                    log.error(line);
                }
            }

            exitCode = process.waitFor();
        } catch (Exception e) {
            log.error("Failed to run DEMIS subprocess.", e);
        }

        if (exitCode == 0) {
            log.info("DEMIS completed successfully!");
            if (openAfterStitching) {
                openStitchedOutput();
            }
        } else {
            log.error("DEMIS failed and exited with code: " + exitCode);
        }

        return exitCode;
    }

    /**
     * Opens all images in the output directory.
     */
    private void openStitchedOutput() {
        // Load paths of all TIFF/PNG/JPEG images from the output directory.
        File[] files = outputDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".tif")
                || name.toLowerCase().endsWith(".tiff")
                || name.toLowerCase().endsWith(".png")
                || name.toLowerCase().endsWith(".jpg")
                || name.toLowerCase().endsWith(".jpeg"));

        if (files == null || files.length == 0) {
            log.info("Could not find any images in the output directory to display.");
            return;
        }

        Arrays.sort(files, Comparator.comparing(File::getName));

        // Display the images in ImageJ.
        for (File f : files) {
            try {
                Object img = ij.io().open(f.getAbsolutePath());
                ij.ui().show(img);
            } catch (Exception e) {
                log.error("Failed to open output image: " + f.getAbsolutePath(), e);
            }
        }
    }

    /**
     * This main function serves for development purposes. It allows to run the
     * plugin immediately out of the integrated development environment (IDE).
     * It creates ImageJ GUI and starts the plugin automatically.
     *
     * @param args Arguments (ignored)
     * @throws Exception
     */
    public static void main(final String... args) throws Exception {
        final ImageJ ij = new ImageJ();
        ij.ui().showUI();
        ij.command().run(DEMIS.class, true);
    }
}
