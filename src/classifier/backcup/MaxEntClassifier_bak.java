package classifier.backcup;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import classifier.maxent.IClassifier;


import opennlp.maxent.BasicEventStream;
import opennlp.maxent.GIS;
import opennlp.maxent.GISModel;
import opennlp.maxent.RealBasicEventStream;
import opennlp.model.ComparablePredicate;
import opennlp.model.Context;
import opennlp.model.DataIndexer;
import opennlp.model.EventStream;
import opennlp.model.IndexHashTable;
import opennlp.model.OnePassDataIndexer;
import opennlp.model.OnePassRealValueDataIndexer;
import opennlp.model.UniformPrior;

//import classifier.maxent.gis.*;

public class MaxEntClassifier_bak implements IClassifier {
    private boolean isReal = false;
    private boolean USE_SMOOTHING = false;
    private double SMOOTHING_OBSERVATION = 0.1;
    private int CUTOFF = 0;
    private int ITERATIONS = 100;

    GISModel model = null;

    Map<String, Double> resultMap = null;

    @Override
    public IClassifier train(List<String[]> trainingdata) {
        GIS.SMOOTHING_OBSERVATION = SMOOTHING_OBSERVATION;
        EventStream es;
        DataIndexer di;

        try {
            if (isReal) {
                es = new RealBasicEventStream(new ListDataStream(trainingdata));
                di = new OnePassRealValueDataIndexer(es, CUTOFF);
            }
            else {
                es = new BasicEventStream(new ListDataStream(trainingdata));
                di = new OnePassDataIndexer(es, CUTOFF);
            }

            model = GIS.trainModel(ITERATIONS, di, true, USE_SMOOTHING, new UniformPrior(), CUTOFF);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return this;
    }

    public void loadModelFromFile(String modelFileName) throws IOException, ClassNotFoundException {

        BufferedReader dataReader = new BufferedReader(new InputStreamReader(new BufferedInputStream(
                new FileInputStream(modelFileName))));

        dataReader.readLine();

        int correctionConstant = Integer.parseInt(dataReader.readLine());
        double correctionParam = Double.parseDouble(dataReader.readLine());

        int numOutcomes = Integer.parseInt(dataReader.readLine());
        String[] outcomeLabels = new String[numOutcomes];
        for (int i = 0; i < numOutcomes; i++)
            outcomeLabels[i] = dataReader.readLine();

        int numOCTypes = Integer.parseInt(dataReader.readLine());
        int[][] outcomePatterns = new int[numOCTypes][];
        for (int i = 0; i < numOCTypes; i++) {
            StringTokenizer tok = new StringTokenizer(dataReader.readLine(), " ");
            int[] infoInts = new int[tok.countTokens()];
            for (int j = 0; tok.hasMoreTokens(); j++) {
                infoInts[j] = Integer.parseInt(tok.nextToken());
            }
            outcomePatterns[i] = infoInts;
        }

        int NUM_PREDS = Integer.parseInt(dataReader.readLine());
        String[] predLabels = new String[NUM_PREDS];
        for (int i = 0; i < NUM_PREDS; i++)
            predLabels[i] = dataReader.readLine();

        Context[] params = new Context[NUM_PREDS];
        int pid = 0;
        for (int i = 0; i < outcomePatterns.length; i++) {
            // construct outcome pattern
            int[] outcomePattern = new int[outcomePatterns[i].length - 1];
            for (int k = 1; k < outcomePatterns[i].length; k++) {
                outcomePattern[k - 1] = outcomePatterns[i][k];
            }
            // System.err.println("outcomePattern "+i+" of "+outcomePatterns.length+" with "+outcomePatterns[i].length+" outcomes ");
            // populate parameters for each context which uses this outcome pattern.
            for (int j = 0; j < outcomePatterns[i][0]; j++) {
                double[] contextParameters = new double[outcomePatterns[i].length - 1];
                for (int k = 1; k < outcomePatterns[i].length; k++) {
                    contextParameters[k - 1] = Double.parseDouble(dataReader.readLine());
                }
                params[pid] = new Context(outcomePattern, contextParameters);
                pid++;
            }
        }

        dataReader.close();
        model = new GISModel(params, predLabels, outcomeLabels, correctionConstant, correctionParam);

        // model = (GISModel) new GenericModelReader(new File(modelFileName)).getModel();
    }

    protected ComparablePredicate[] sortValues(Context[] PARAMS, String[] PRED_LABELS) {

        ComparablePredicate[] sortPreds = new ComparablePredicate[PARAMS.length];

        for (int pid = 0; pid < PARAMS.length; pid++) {
            int[] predkeys = PARAMS[pid].getOutcomes();
            int[] activeOutcomes = predkeys;
            double[] activeParams = PARAMS[pid].getParameters();

            sortPreds[pid] = new ComparablePredicate(PRED_LABELS[pid], activeOutcomes, activeParams);
        }

        Arrays.sort(sortPreds);
        return sortPreds;
    }

    @SuppressWarnings({ "rawtypes", "unchecked" })
    protected List<ComparablePredicate> compressOutcomes(ComparablePredicate[] sorted) {
        ComparablePredicate cp = sorted[0];
        List outcomePatterns = new ArrayList();
        List newGroup = new ArrayList();
        for (int i = 0; i < sorted.length; i++) {
            if (cp.compareTo(sorted[i]) == 0) {
                newGroup.add(sorted[i]);
            }
            else {
                cp = sorted[i];
                outcomePatterns.add(newGroup);
                newGroup = new ArrayList();
                newGroup.add(sorted[i]);
            }
        }
        outcomePatterns.add(newGroup);
        return outcomePatterns;
    }

    public void persistModel(String modelFileName) throws IOException {

        BufferedWriter bw = new BufferedWriter(new FileWriter(new File(modelFileName)));

        Object[] data = model.getDataStructures();

        Context[] PARAMS = (Context[]) data[0];
        @SuppressWarnings("unchecked")
        IndexHashTable<String> pmap = (IndexHashTable<String>) data[1];
        String[] OUTCOME_LABELS = (String[]) data[2];
        int CORRECTION_CONSTANT = ((Integer) data[3]).intValue();
        double CORRECTION_PARAM = ((Double) data[4]).doubleValue();

        String[] PRED_LABELS = new String[pmap.size()];
        pmap.toArray(PRED_LABELS);

        // the type of model (GIS)
        bw.write("GIS");
        bw.newLine();

        // the value of the correction constant
        bw.write(Integer.toString(CORRECTION_CONSTANT));
        bw.newLine();

        // the value of the correction constant
        bw.write(Double.toString(CORRECTION_PARAM));
        bw.newLine();

        // the mapping from outcomes to their integer indexes
        bw.write(Integer.toString(OUTCOME_LABELS.length));
        bw.newLine();

        for (int i = 0; i < OUTCOME_LABELS.length; i++) {
            bw.write(OUTCOME_LABELS[i]);
            bw.newLine();
        }

        // the mapping from predicates to the outcomes they contributed to.
        // The sorting is done so that we actually can write this out more
        // compactly than as the entire list.
        ComparablePredicate[] sorted = sortValues(PARAMS, PRED_LABELS);
        List<ComparablePredicate> compressed = compressOutcomes(sorted);

        bw.write(Integer.toString(compressed.size()));
        bw.newLine();

        for (int i = 0; i < compressed.size(); i++) {
            @SuppressWarnings("unchecked")
            List<ComparablePredicate> a = (List<ComparablePredicate>) compressed.get(i);
            bw.write(a.size() + ((ComparablePredicate) a.get(0)).toString());
            bw.newLine();
        }

        // the mapping from predicate names to their integer indexes
        bw.write(Integer.toString(PARAMS.length));
        bw.newLine();

        for (int i = 0; i < sorted.length; i++) {
            bw.write(sorted[i].name);
            bw.newLine();
        }

        // write out the parameters
        for (int i = 0; i < sorted.length; i++)
            for (int j = 0; j < sorted[i].params.length; j++) {
                bw.write(Double.toString(sorted[i].params[j]));
                bw.newLine();
            }

        bw.flush();
        bw.close();

    }

    // Predict only single feature vector.
    @Override
    public Map<String, Double> predict(String[] featureVector) {
        if (model == null) throw new RuntimeException("Model is null.");
        double[] ocs;
        float[] realVals = null;

        if (isReal) {
            boolean hasReal = false;
            realVals = new float[featureVector.length];
            for (int i = 0; i < featureVector.length; i++) {
                int loc = featureVector[i].lastIndexOf("=") + 1;
                if (loc == 0 || loc == featureVector[i].length()) realVals[i] = 1;
                else {
                    hasReal = true;
                    realVals[i] = Float.parseFloat(featureVector[i].substring(loc));
                    featureVector[i] = featureVector[i].substring(0, loc - 1);
                    if (realVals[i] < 0) {
                        throw new RuntimeException("Negitive values are not allowed: " + realVals[i]);
                    }
                }
            }

            if (!hasReal) realVals = null;
        }

        ocs = model.eval(featureVector, realVals);

        // <Group Tag, Probability>
        Map<String, Double> result = new HashMap<String, Double>();
        for (int i = 0; i < model.getNumOutcomes(); i++) {
            result.put(model.getOutcome(i), ocs[i]);
        }

        resultMap = result;

        return resultMap;
    }

    public String getResult() {
        if (resultMap == null) throw new RuntimeException("Predicting First");

        double max = 0;
        String maxTag = null;

        for (String key : resultMap.keySet()) {
            if (resultMap.get(key) > max) {
                max = resultMap.get(key);
                maxTag = key;
            }
        }

        return maxTag;
    }

    public MaxEntClassifier_bak(boolean real, boolean usesmoothing, double smoothingObv) {
        this.isReal = real;
        this.USE_SMOOTHING = usesmoothing;
        this.SMOOTHING_OBSERVATION = smoothingObv;
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        List<String[]> traindata = new ArrayList<String[]>();
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.5", "win" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.5", "win" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.6666", "lose" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.3333", "win" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.6666", "win" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.3333", "win" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.75", "win" });
        traindata.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.25", "tie" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.5", "tie" });
        traindata.add(new String[] { "away", "pdiff=0.9375", "ptwins=0.25", "tie" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.6", "tie" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.25", "tie" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.25", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.6", "lose" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.25", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.6", "win" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.4", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.6666", "lose" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.4", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.5", "tie" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.7142", "win" });
        traindata.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.5", "win" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5714", "win" });
        traindata.add(new String[] { "away", "pdiff=0.9375", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.625", "win" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.4285", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.5714", "win" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.5555", "lose" });
        traindata.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.5555", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.9375", "ptwins=0.5", "tie" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.6", "win" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.5555", "win" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.6", "tie" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.5", "win" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.5454", "win" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.5", "win" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.6", "win" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.4444", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.4545", "tie" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5454", "tie" });
        traindata.add(new String[] { "away", "pdiff=0.9375", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.5384", "tie" });
        traindata.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.4545", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5454", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.5454", "win" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.5384", "lose" });
        traindata.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5833", "win" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.5714", "lose" });
        traindata.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.5384", "win" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.5384", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.9375", "ptwins=0.5384", "win" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.6", "tie" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.5", "tie" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.5714", "win" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.5", "win" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.6", "lose" });
        traindata.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5333", "win" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.4666", "win" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.625", "lose" });
        traindata.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.5333", "tie" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.5", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.4375", "win" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.6470", "win" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.5333", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5294", "tie" });
        traindata.add(new String[] { "away", "pdiff=0.9375", "ptwins=0.4117", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.6111", "tie" });
        traindata.add(new String[] { "away", "pdiff=1.0625", "ptwins=0.5625", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5294", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.9375", "ptwins=0.4444", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.6111", "lose" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.5882", "tie" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5555", "win" });
        traindata.add(new String[] { "away", "pdiff=0.9375", "ptwins=0.4736", "tie" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.6315", "win" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.5882", "tie" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5263", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.4736", "win" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.6", "lose" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.5882", "tie" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.55", "tie" });
        traindata.add(new String[] { "home", "pdiff=0.9375", "ptwins=0.45", "win" });
        traindata.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.6190", "lose" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.5882", "tie" });
        traindata.add(new String[] { "away", "pdiff=0.8125", "ptwins=0.55", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.9375", "ptwins=0.4285", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.6875", "ptwins=0.6363", "lose" });
        traindata.add(new String[] { "home", "pdiff=1.0625", "ptwins=0.5882", "lose" });
        traindata.add(new String[] { "home", "pdiff=0.8125", "ptwins=0.5714", "lose" });
        traindata.add(new String[] { "away", "pdiff=0.9375", "ptwins=0.4545", "lose" });

        MaxEntClassifier_bak maxent = new MaxEntClassifier_bak(true, false, 0);
        maxent.train(traindata);
        String modelFile = "fixture/model.txt";
        maxent.persistModel(modelFile);
        maxent.loadModelFromFile(modelFile);

        List<String[]> predictData = new ArrayList<String[]>();
        predictData.add(new String[] { "home", "pdiff=0.6875", "ptwins=0.5" });

        for (String[] feature : predictData) {
            System.out.println(Arrays.toString(feature));
            System.out.println(maxent.predict(feature));
            System.out.println(maxent.getResult());
        }

    }
}
