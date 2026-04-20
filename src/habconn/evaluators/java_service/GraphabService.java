/**
 * Persistent Java service for exact incremental patch addition over Graphab internals.
 *
 * Protocol: JSON-over-stdin/stdout, one command per line, one response per line.
 *
 * Commands:
 *   {"cmd": "create_session", "project_dir": "...", "linkset_name": "...", "graph_name": "...",
 *    "metric_name": "PC", "metric_d": 13.0, "metric_p": 0.01}
 *   {"cmd": "add_patch", "wkt": "POLYGON(...)", "capacity": 1.0, "restored_resistance": 1.0}
 *   {"cmd": "compute_pc"}
 *   {"cmd": "reset_session"}
 *   {"cmd": "shutdown"}
 *
 * Responses:
 *   {"status": "ok", ...}
 *   {"status": "error", "message": "..."}
 *
 * This service supports exact additive patch addition with resistance restoration.
 * When a patch is added, both the habitat (source) raster and the cost (resistance)
 * raster are updated before computing new links. This matches the CLI's behavior
 * of burning patches into both rasters.
 */

import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.lang.ref.SoftReference;
import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.Envelope;
import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.geom.GeometryFactory;
import org.locationtech.jts.geom.Point;
import org.locationtech.jts.geom.util.AffineTransformation;
import org.locationtech.jts.io.WKTReader;
import org.thema.data.feature.DefaultFeature;
import org.thema.graphab.Project;
import org.thema.graphab.graph.GraphGenerator;
import org.thema.graphab.links.Linkset;
import org.thema.graphab.metric.global.GlobalMetricLauncher;
import org.thema.graphab.metric.global.PCMetric;

public class GraphabService {

    private static final Logger LOG = Logger.getLogger(GraphabService.class.getName());

    private Project project;
    private Linkset linkset;
    private GraphGenerator baseGraph;
    private GlobalMetricLauncher metricLauncher;

    // Session configuration for reset
    private File sessionProjectDir;
    private String sessionLinksetName;
    private String sessionGraphName;
    private double sessionMetricD;
    private double sessionMetricP;

    // Writable copy of the cost raster for resistance updates
    private WritableRaster costRaster;

    // Track added patches for reporting
    private int patchesAdded = 0;

    public static void main(String[] args) throws Exception {
        // Suppress Graphab GUI logger noise
        Logger.getLogger("org.thema").setLevel(Level.WARNING);

        GraphabService service = new GraphabService();

        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        PrintWriter out = new PrintWriter(System.out, true);

        // Signal readiness
        out.println("{\"status\":\"ready\",\"version\":\"1.0\"}");

        String line;
        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;

            try {
                String response = service.handleCommand(line);
                out.println(response);
                if (line.contains("\"shutdown\"")) {
                    break;
                }
            } catch (Exception e) {
                String msg = e.getMessage() != null ? e.getMessage() : e.getClass().getName();
                out.println(jsonError(msg.replace("\"", "'")));
            }
        }
    }

    private String handleCommand(String json) throws Exception {
        String cmd = extractString(json, "cmd");

        switch (cmd) {
            case "create_session":
                return handleCreateSession(json);
            case "add_patch":
                return handleAddPatch(json);
            case "compute_pc":
                return handleComputePC();
            case "reset_session":
                return handleResetSession();
            case "shutdown":
                return "{\"status\":\"ok\",\"message\":\"shutting down\"}";
            default:
                return jsonError("unknown command: " + cmd);
        }
    }

    private String handleCreateSession(String json) throws Exception {
        String projectDir = extractString(json, "project_dir");
        String linksetName = extractString(json, "linkset_name");
        String graphName = extractString(json, "graph_name");
        double metricD = extractDouble(json, "metric_d", 130.0);
        double metricP = extractDouble(json, "metric_p", 0.01);

        this.sessionProjectDir = new File(projectDir);
        this.sessionLinksetName = linksetName;
        this.sessionGraphName = graphName;
        this.sessionMetricD = metricD;
        this.sessionMetricP = metricP;

        loadSession();

        int nPatches = project.getPatches().size();

        return String.format(
            "{\"status\":\"ok\",\"n_patches\":%d,\"patches_added\":0,\"project_dir\":\"%s\"}",
            nPatches, projectDir.replace("\\", "/")
        );
    }

    private void loadSession() throws Exception {
        File xmlFile = findProjectXml(this.sessionProjectDir);
        this.project = Project.loadProject(xmlFile, true);
        this.linkset = project.getLinkset(this.sessionLinksetName);

        // Find existing graph
        GraphGenerator existingGraph = null;
        for (GraphGenerator g : project.getGraphs()) {
            if (g.getName().equals(this.sessionGraphName)) {
                existingGraph = g;
                break;
            }
        }
        if (existingGraph == null) {
            throw new IllegalStateException(
                "Graph '" + this.sessionGraphName + "' not found in project"
            );
        }
        this.baseGraph = existingGraph;

        // Set up PC metric (d is already in Euclidean units, converted by Python)
        PCMetric pcMetric = new PCMetric();
        HashMap<String, Object> params = new HashMap<>();
        params.put("d", Double.valueOf(this.sessionMetricD));
        params.put("p", Double.valueOf(this.sessionMetricP));
        pcMetric.setParams(params);
        this.metricLauncher = new GlobalMetricLauncher(pcMetric);

        // Prepare a writable copy of the cost raster for resistance updates.
        // The cost raster is cached in Project.extRasters. We create a writable
        // copy and inject it back so that pathfinders use our modified version.
        this.costRaster = prepareWritableCostRaster();

        System.err.println("[GraphabService] Session loaded: d=" + this.sessionMetricD
            + " p=" + this.sessionMetricP + " patches=" + project.getPatches().size());

        this.patchesAdded = 0;
    }

    /**
     * Get the external cost raster, create a writable copy, and inject it
     * into the project's cache so subsequent pathfinder creations use it.
     */
    @SuppressWarnings("unchecked")
    private WritableRaster prepareWritableCostRaster() throws Exception {
        // Get the ext cost file from the linkset
        File extCostFile = linkset.getExtCostFile();
        if (extCostFile == null) {
            System.err.println("[GraphabService] No external cost file, skipping cost raster setup");
            return null;
        }

        // Load the raster through Project's normal path.
        // Note: getExtRaster returns a raster with minX=1,minY=1 offset
        // (via createTranslatedChild(1,1)). We must preserve this offset.
        Raster originalRaster = project.getExtRaster(extCostFile);
        int minX = originalRaster.getMinX();
        int minY = originalRaster.getMinY();
        int w = originalRaster.getWidth();
        int h = originalRaster.getHeight();

        // Create a writable copy with the same offset
        WritableRaster writable = Raster.createWritableRaster(
            originalRaster.getSampleModel(),
            new java.awt.Point(minX, minY)
        );
        // Copy pixel data
        for (int y = minY; y < minY + h; y++) {
            for (int x = minX; x < minX + w; x++) {
                writable.setSample(x, y, 0, originalRaster.getSampleDouble(x, y, 0));
            }
        }

        // Inject the writable raster into Project.extRasters cache via reflection
        // so that getPathFinder() will use our modified version
        injectCostRaster(extCostFile, writable);

        return writable;
    }

    /**
     * Inject a raster into Project.extRasters cache using reflection.
     */
    @SuppressWarnings("unchecked")
    private void injectCostRaster(File extCostFile, Raster raster) throws Exception {
        Field extRastersField = Project.class.getDeclaredField("extRasters");
        extRastersField.setAccessible(true);
        Object extRasters = extRastersField.get(project);
        if (extRasters == null) {
            extRasters = new HashMap<File, Object>();
            extRastersField.set(project, extRasters);
        }
        // The cache stores SoftReference-like objects (Project.SoftRef)
        // Find the inner class and create an instance
        Class<?>[] innerClasses = Project.class.getDeclaredClasses();
        Class<?> softRefClass = null;
        for (Class<?> c : innerClasses) {
            if (c.getSimpleName().equals("SoftRef")) {
                softRefClass = c;
                break;
            }
        }

        if (softRefClass != null) {
            // Use Project.SoftRef constructor
            Object ref = softRefClass.getDeclaredConstructors()[0].newInstance(raster);
            ((Map<File, Object>) extRasters).put(extCostFile, ref);
        } else {
            // Fallback: try StrongRef
            for (Class<?> c : innerClasses) {
                if (c.getSimpleName().equals("StrongRef")) {
                    softRefClass = c;
                    break;
                }
            }
            if (softRefClass != null) {
                Object ref = softRefClass.getDeclaredConstructors()[0].newInstance(raster);
                ((Map<File, Object>) extRasters).put(extCostFile, ref);
            }
        }
    }

    private String handleAddPatch(String json) throws Exception {
        if (project == null) {
            return jsonError("no active session");
        }

        String wkt = extractString(json, "wkt");
        double capacity = extractDouble(json, "capacity", 1.0);
        double restoredResistance = extractDouble(json, "restored_resistance", -1.0);

        WKTReader wktReader = new WKTReader();
        Geometry geom = wktReader.read(wkt);

        // Add patch to project (updates habitat/source raster)
        DefaultFeature patch = project.addPatch(geom, capacity);

        // Also update the cost raster for the patch area to restored resistance
        if (costRaster != null && restoredResistance > 0) {
            updateCostRaster(geom, restoredResistance);
        }

        // Compute new links (will use updated cost raster via pathfinder)
        linkset.addLinks(patch);

        patchesAdded++;
        int patchId = (Integer) patch.getId();

        return String.format(
            "{\"status\":\"ok\",\"patch_id\":%d,\"patches_added\":%d}",
            patchId, patchesAdded
        );
    }

    /**
     * Update the cost raster pixels under the given geometry to the specified
     * resistance value. This matches the CLI's behavior of burning restored
     * resistance values into the resistance raster for selected planning units.
     */
    private void updateCostRaster(Geometry geom, double resistanceValue) throws Exception {
        AffineTransformation space2grid = project.getSpace2grid();

        if (geom instanceof Point) {
            Coordinate cg = space2grid.transform(
                geom.getCoordinate(), new Coordinate()
            );
            int x = (int) cg.x;
            int y = (int) cg.y;
            if (x >= costRaster.getMinX() && x < costRaster.getMinX() + costRaster.getWidth()
                && y >= costRaster.getMinY() && y < costRaster.getMinY() + costRaster.getHeight()) {
                costRaster.setSample(x, y, 0, resistanceValue);
            }
        } else {
            // Polygon: iterate over pixels inside the geometry (same as Project.addPatch)
            Geometry geomGrid = space2grid.transform(geom);
            Envelope env = geomGrid.getEnvelopeInternal();
            GeometryFactory geomFactory = geom.getFactory();

            for (double y = (double)((int)env.getMinY()) + 0.5; y <= Math.ceil(env.getMaxY()); y += 1.0) {
                for (double x = (double)((int)env.getMinX()) + 0.5; x <= Math.ceil(env.getMaxX()); x += 1.0) {
                    Point p = geomFactory.createPoint(new Coordinate(x, y));
                    if (!geomGrid.contains(p)) continue;
                    int ix = (int) x;
                    int iy = (int) y;
                    if (ix >= costRaster.getMinX() && ix < costRaster.getMinX() + costRaster.getWidth()
                        && iy >= costRaster.getMinY() && iy < costRaster.getMinY() + costRaster.getHeight()) {
                        costRaster.setSample(ix, iy, 0, resistanceValue);
                    }
                }
            }
        }
    }

    private String handleComputePC() throws Exception {
        if (project == null) {
            return jsonError("no active session");
        }

        // Create fresh graph copy for metric calculation (same pattern as AddPatchCommand)
        GraphGenerator freshGraph = new GraphGenerator(this.baseGraph, "");
        Double[] results = metricLauncher.calcMetric(freshGraph, false, null);
        double pc = results[0];

        return String.format(
            "{\"status\":\"ok\",\"pc_value\":%.20e,\"patches_added\":%d}",
            pc, patchesAdded
        );
    }

    private String handleResetSession() throws Exception {
        if (sessionProjectDir == null) {
            return jsonError("no session to reset");
        }

        loadSession();

        return String.format(
            "{\"status\":\"ok\",\"patches_added\":0,\"n_patches\":%d}",
            project.getPatches().size()
        );
    }

    // --- Utility methods ---

    private static File findProjectXml(File projectDir) {
        File[] xmlFiles = projectDir.listFiles((dir, name) -> name.endsWith(".xml"));
        if (xmlFiles == null || xmlFiles.length == 0) {
            throw new IllegalArgumentException(
                "No .xml project file found in: " + projectDir.getAbsolutePath()
            );
        }
        return xmlFiles[0];
    }

    private static String jsonError(String message) {
        return "{\"status\":\"error\",\"message\":\"" + message + "\"}";
    }

    private static String extractString(String json, String key) {
        String pattern = "\"" + key + "\"";
        int idx = json.indexOf(pattern);
        if (idx < 0) return "";

        idx = json.indexOf(":", idx + pattern.length());
        if (idx < 0) return "";

        int start = json.indexOf("\"", idx + 1);
        if (start < 0) return "";

        int end = start + 1;
        while (end < json.length()) {
            if (json.charAt(end) == '\\') {
                end += 2;
                continue;
            }
            if (json.charAt(end) == '"') break;
            end++;
        }

        return json.substring(start + 1, end);
    }

    private static double extractDouble(String json, String key, double defaultValue) {
        String pattern = "\"" + key + "\"";
        int idx = json.indexOf(pattern);
        if (idx < 0) return defaultValue;

        idx = json.indexOf(":", idx + pattern.length());
        if (idx < 0) return defaultValue;

        int start = idx + 1;
        while (start < json.length() && Character.isWhitespace(json.charAt(start))) {
            start++;
        }

        int end = start;
        while (end < json.length() && (Character.isDigit(json.charAt(end))
                || json.charAt(end) == '.' || json.charAt(end) == '-'
                || json.charAt(end) == 'e' || json.charAt(end) == 'E'
                || json.charAt(end) == '+')) {
            end++;
        }

        if (start == end) return defaultValue;
        return Double.parseDouble(json.substring(start, end));
    }
}
