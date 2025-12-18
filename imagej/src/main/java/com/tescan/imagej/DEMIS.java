/**
 * Deep Electron Microscopy Image Stitching (DEMIS) runner plugin for ImageJ.
 * 
 * @author Petr Šilling
 * @year 2025
 */

package com.tescan.imagej;

import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imglib2.type.numeric.RealType;

import org.scijava.ItemVisibility;
import org.scijava.command.Command;
import org.scijava.io.IOService;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
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
 * The plugin works by running DEMIS CLI in the background and importing the result. 
 */
@Plugin(type = Command.class, menuPath = "Plugins>Stitching>DEMIS")
public class DEMIS<T extends RealType<T>> implements Command {
	@Parameter(label = "DEMIS Script Paths", visibility = ItemVisibility.MESSAGE)
    private String demisPathsLabel = "";
	
	@Parameter(label = " ", visibility = ItemVisibility.MESSAGE)
    private String demisPathsLabelSpace = "";
	
	@Parameter(label = "DEMIS root directory", style = "directory")
    private File demisDir;
    
    @Parameter(label = "Python executable", style = "file")
    private File pythonExe;

    @Parameter(label = "LoFTR checkpoint path", required = false, style = "file")
    private File loftrCheckpointPath = null;
    
    @Parameter(label = "YAML config file", description = "If omitted, the parameters below will be used.", required = false, style = "file")
    private File yamlConfig = null;
	
    @Parameter(label = "Dataset Settings", visibility = ItemVisibility.MESSAGE)
    private String datasetSettingsLabel = "";
	
    @Parameter(label = " ", visibility = ItemVisibility.MESSAGE)
    private String datasetSettingsLabelSpace = "";
    
    @Parameter(label = "Source dataset directory", style = "directory")
    private File inputDir;

    @Parameter(label = "Output directory", style = "directory")
    private File outputDir;

    @Parameter(label = "Open output files after stitching")
    private Boolean openAfterStitching = true;
    
    @Parameter(label = "Grid rows", min = "1", required = false)
    private Integer gridRows = 0;

    @Parameter(label = "Grid columns", min = "1", required = false)
    private Integer gridCols = 0;
	
	@Parameter(label = "Stitching Parameters (only used if config is empty)", visibility = ItemVisibility.MESSAGE)
	private String stitchingParametersLabel = "";
	
	@Parameter(label = " ", visibility = ItemVisibility.MESSAGE)
	private String stitchingParametersLabelSpace = "";
	
	@Parameter(label = "Tile overlap", required = false, min = "0.01", max = "0.99", stepSize = "0.01")
    private Double tileOverlap = 0.3;

    @Parameter(label = "Resolution ratio", required = false, min = "0.1", stepSize = "0.1")
    private Double resolutionRatio = 0.5;

    @Parameter(label = "Matching method", required = false, choices = {"loftr", "sift", "orb"})
    private String matchingMethod = "loftr";

    @Parameter(label = "Transform type", required = false,
               choices = {"translation", "euclidean", "similarity", "affine", "projective"})
    private String transformType = "euclidean";

    @Parameter(label = "Construction method", required = false, choices = {"optimised", "mst", "slam"})
    private String constructionMethod = "optimised";

    @Parameter(label = "Compositing method", required = false, choices = {"overwrite", "average", "adaptive"})
    private String compositingMethod = "overwrite";

    @Parameter(label = "Optical Flow Refinement", required = false)
    private Boolean opticalFlowRefinement = true;

    @Parameter(label = "Optical Flow Refinement Type", required = false, choices = {"grid", "mean"})
    private String opticalFlowRefinementType = "grid";

    @Parameter(label = "Normalise intensity", required = false)
    private Boolean normaliseIntensity = true;
    
    @Parameter(label = "Colored output", required = false)
    private Boolean coloredOutput = false;
	
    /** ImageJ */
    @Parameter
    private ImageJ ij;
	
    /** ImageJ IO service. */
    @Parameter
    private IOService io;
    
    /** ImageJ UI service. */
    @Parameter
    private UIService ui;

    /** ImageJ logging service. */
    @Parameter
    private LogService log;

    /**
     * Entry point of the DEMIS CLI plugin.
     */
    @Override
    public void run() {

        // Get rows and columns.
        parseGridInfo();

        // Validate grid sizes.
        if (gridRows == null || gridCols == null) {
            log.error("Grid rows/cols are invalid.");
            return;
        }
        
        List<String> cmd = buildDEMISCmd();
        runSticher(cmd);
    }

    /**
     * Try to parse grid size from the input directory name. The expected format is <rows>x<cols>.
     */
    private void parseGridInfo() {
        if (gridRows == null || gridCols == null) {
            String name = inputDir.getName();
            String[] parts = name.split("[xX]");
            if (parts.length == 2) {
                try {
                    gridRows = Integer.parseInt(parts[0]);
                    gridCols = Integer.parseInt(parts[1]);
                } catch (NumberFormatException e) {
                	log.error("Failed to parse grid sizes from input directory name", e);
                }
            }
        }
    }
    
    /**
     * Build the command for DEMIS CLI using the given parameters.
     * @return The command as a list of strings.
     */
    private List<String> buildDEMISCmd() {
    	List<String> cmd = new ArrayList<>();
    	Path pathToStitchScript = Paths.get(demisDir.getAbsolutePath(), "scripts/stitch.py"); 
        
        cmd.add(pythonExe.getAbsolutePath());
        cmd.add(pathToStitchScript.toAbsolutePath().toString());
        
        cmd.add("DATASET.PATH");
        cmd.add(inputDir.getAbsolutePath());

        cmd.add("STITCHER.OUTPUT_PATH");
        cmd.add(outputDir.getAbsolutePath());
        
        if (yamlConfig != null && yamlConfig.exists()) {
        	// If using YAML config, pass only the config.
            cmd.add("--config");
            cmd.add(yamlConfig.getAbsolutePath());
        } else {
            // No config, build the full command using all optional stitching parameters.
        	cmd.add("DATASET.ROWS");
    		cmd.add(gridRows.toString());
        	
            cmd.add("DATASET.COLS");
            cmd.add(gridCols.toString());

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
        
        return cmd;
    }
    

    
    /**
     * Runs the DEMIS stitcher in a subprocess, logging all output.
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
            ij.command().run("Log", true);
            
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
    	File[] files = outputDir.listFiles((dir, name) ->
        	name.toLowerCase().endsWith(".tif")
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
        		Object img = io.open(f.getAbsolutePath());
        		ui.show(img);
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
