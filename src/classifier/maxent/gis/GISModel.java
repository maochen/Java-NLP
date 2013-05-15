package classifier.maxent.gis;

import java.io.Serializable;
import java.util.Arrays;

/**
 * A maximum entropy model which has been trained using the Generalized Iterative Scaling procedure
 * (implemented in GIS.java).
 * 
 * @author Tom Morton and Jason Baldridge
 * @version $Revision: 1.6 $, $Date: 2010/09/06 08:02:18 $
 */
public final class GISModel implements Serializable {

    /**
     * The {@link IndexHashTable} is a hash table which maps entries of an array to their index in
     * the array. All entries in the array must be unique otherwise a well-defined mapping is not
     * possible.
     * <p>
     * The entry objects must implement {@link Object#equals(Object)} and {@link Object#hashCode()}
     * otherwise the behavior of this class is undefined.
     * <p>
     * The implementation uses a hash table with open addressing and linear probing.
     * <p>
     * The table is thread safe and can concurrently accessed by multiple threads, thread safety is
     * achieved through immutability. Though its not strictly immutable which means, that the table
     * must still be safely published to other threads.
     */
    static class IndexHashTable<T> implements Serializable{

        private final Object keys[];
        private final int values[];

        /**
         * Initializes the current instance. The specified array is copied into the table and later
         * changes to the array do not affect this table in any way.
         * 
         * @param mapping the values to be indexed, all values must be unique otherwise a
         *            well-defined mapping of an entry to an index is not possible
         * @param loadfactor the load factor, usually 0.7
         * 
         * @throws IllegalArgumentException if the entries are not unique
         */
        public IndexHashTable(T mapping[], double loadfactor) {
            if (loadfactor <= 0 || loadfactor > 1)
                throw new IllegalArgumentException("loadfactor must be larger than 0 "
                        + "and equal to or smaller than 1!");

            int arraySize = (int) (mapping.length / loadfactor) + 1;

            keys = new Object[arraySize];
            values = new int[arraySize];

            for (int i = 0; i < mapping.length; i++) {
                int startIndex = (mapping[i].hashCode() & 0x7fffffff) % keys.length;
                // indexForHash(mapping[i].hashCode(), keys.length);

                int index = searchKey(startIndex, null, true);

                if (index == -1) throw new IllegalArgumentException("Array must contain only unique keys!");

                keys[index] = mapping[i];
                values[index] = i;
            }
        }

        private int searchKey(int startIndex, Object key, boolean insert) {

            for (int index = startIndex; true; index = (index + 1) % keys.length) {

                // The keys array contains at least one null element, which guarantees
                // termination of the loop
                if (keys[index] == null) {
                    if (insert) return index;
                    else return -1;
                }

                if (keys[index].equals(key)) {
                    if (!insert) return index;
                    else return -1;
                }
            }
        }

        /**
         * Retrieves the index for the specified key.
         * 
         * @param key
         * @return the index or -1 if there is no entry to the keys
         */
        public int get(T key) {

            int startIndex = (key.hashCode() & 0x7fffffff) % keys.length;
            // indexForHash(key.hashCode(), keys.length);
            int index = searchKey(startIndex, key, false);

            return (index != -1) ? values[index] : -1;
        }

        // /**
        // * Retrieves the size.
        // *
        // * @return the number of elements in this map.
        // */
        // public int size() {
        // return size;
        // }
        //
        // @SuppressWarnings("unchecked")
        // public T[] toArray(T array[]) {
        // for (int i = 0; i < keys.length; i++) {
        // if (keys[i] != null)
        // array[values[i]] = (T) keys[i];
        // }
        //
        // return array;
        // }
    }

    /**
     * Class which associates a real valued parameter or expected value with a particular contextual
     * predicate or feature. This is used to store maxent model parameters as well as model and
     * empirical expected values.
     * 
     */
    static class Context implements Serializable{

        /** The real valued parameters or expected values for this context. */
        protected double[] parameters;
        /** The outcomes which occur with this context. */
        protected int[] outcomes;

        /**
         * Creates a new parameters object with the specified parameters associated with the
         * specified outcome pattern.
         * 
         * @param outcomePattern Array of outcomes for which parameters exists for this context.
         * @param parameters Parameters for the outcomes specified.
         */
        public Context(int[] outcomePattern, double[] parameters) {
            this.outcomes = outcomePattern;
            this.parameters = parameters;
        }

        public boolean contains(int outcome) {
            return (Arrays.binarySearch(outcomes, outcome) >= 0);
        }

    }

    static class EvalParameters implements Serializable{

        /**
         * Mapping between outcomes and paramater values for each context. The integer
         * representation of the context can be found using <code>pmap</code>.
         */
        private Context[] params;
        /** The number of outcomes being predicted. */
        private final int numOutcomes;
        /**
         * The maximum number of feattures fired in an event. Usually refered to a C. This is used
         * to normalize the number of features which occur in an event.
         */
        private double correctionConstant;

        /** Stores inverse of the correction constant, 1/C. */
        private final double constantInverse;
        /** The correction parameter of the model. */
        private double correctionParam;

        /**
         * Creates a set of paramters which can be evaulated with the eval method.
         * 
         * @param params The parameters of the model.
         * @param correctionParam The correction paramter.
         * @param correctionConstant The correction constant.
         * @param numOutcomes The number of outcomes.
         */
        public EvalParameters(Context[] params, double correctionParam, double correctionConstant, int numOutcomes) {
            this.params = params;
            this.correctionParam = correctionParam;
            this.numOutcomes = numOutcomes;
            this.correctionConstant = correctionConstant;
            this.constantInverse = 1.0 / correctionConstant;
        }

        /*
         * (non-Javadoc)
         * 
         * @see opennlp.model.EvalParameters#getParams()
         */
        public Context[] getParams() {
            return params;
        }

        /*
         * (non-Javadoc)
         * 
         * @see opennlp.model.EvalParameters#getNumOutcomes()
         */
        public int getNumOutcomes() {
            return numOutcomes;
        }

        public double getCorrectionConstant() {
            return correctionConstant;
        }

        public double getConstantInverse() {
            return constantInverse;
        }

        public double getCorrectionParam() {
            return correctionParam;
        }

        public void setCorrectionParam(double correctionParam) {
            this.correctionParam = correctionParam;
        }
    }

    /** Mapping between predicates/contexts and an integer representing them. */
    protected IndexHashTable<String> pmap;
    /** The names of the outcomes. */
    protected String[] labels;
    /** Parameters for the model. */
    protected EvalParameters evalParams;

    /** Prior distribution for this model. */
    protected UniformPrior prior;

    /** The type of the model. */
    protected ModelType modelType;

    public enum ModelType {
        Maxent, Perceptron
    };

    /**
     * Creates a new model with the specified parameters, outcome names, and predicate/feature
     * labels.
     * 
     * @param params The parameters of the model.
     * @param predLabels The names of the predicates used in this model.
     * @param outcomeNames The names of the outcomes this model predicts.
     * @param correctionConstant The maximum number of active features which occur in an event.
     * @param correctionParam The parameter associated with the correction feature.
     * @param prior The prior to be used with this model.
     */
    public GISModel(Context[] params, String[] predLabels, String[] outcomeNames, int correctionConstant,
            double correctionParam, UniformPrior prior) {
        this.pmap = new IndexHashTable<String>(predLabels, 0.7d);
        this.labels = outcomeNames;
        this.evalParams = new EvalParameters(params, correctionParam, correctionConstant, outcomeNames.length);

        this.prior = prior;
        prior.setLabels(outcomeNames, predLabels);
        modelType = ModelType.Maxent;
    }

    /**
     * Use this model to evaluate a context and return an array of the likelihood of each outcome
     * given that context.
     * 
     * @param context The names of the predicates which have been observed at the present decision
     *            point.
     * @return The normalized probabilities for the outcomes given the context. The indexes of the
     *         double[] are the outcome ids, and the actual string representation of the outcomes
     *         can be obtained from the method getOutcome(int i).
     */

    /**
     * Use this model to evaluate a context and return an array of the likelihood of each outcome
     * given that context.
     * 
     * @param featureVector The names of the predicates which have been observed at the present
     *            decision point.
     * @param outsums This is where the distribution is stored.
     * @return The normalized probabilities for the outcomes given the context. The indexes of the
     *         double[] are the outcome ids, and the actual string representation of the outcomes
     *         can be obtained from the method getOutcome(int i).
     */
    public final double[] eval(String[] featureVector, float[] values) {
        double[] outsums = new double[evalParams.getNumOutcomes()];

        int[] scontexts = new int[featureVector.length];
        for (int i = 0; i < featureVector.length; i++) {
            Integer ci = pmap.get(featureVector[i]);
            scontexts[i] = ci == null ? -1 : ci;
        }

        prior.logPrior(outsums, scontexts);
        return GISModel.eval(scontexts, values, outsums, evalParams);
    }

    /**
     * Use this model to evaluate a context and return an array of the likelihood of each outcome
     * given the specified context and the specified parameters.
     * 
     * @param context The integer values of the predicates which have been observed at the present
     *            decision point.
     * @param values The values for each of the parameters.
     * @param prior The prior distribution for the specified context.
     * @param model The set of parametes used in this computation.
     * @return The normalized probabilities for the outcomes given the context. The indexes of the
     *         double[] are the outcome ids, and the actual string representation of the outcomes
     *         can be obtained from the method getOutcome(int i).
     */
    public static double[] eval(int[] context, float[] values, double[] prior, EvalParameters model) {
        Context[] params = model.getParams();
        int numfeats[] = new int[model.getNumOutcomes()];
        int[] activeOutcomes;
        double[] activeParameters;
        double value = 1;
        for (int ci = 0; ci < context.length; ci++) {
            if (context[ci] >= 0) {
                Context predParams = params[context[ci]];
                activeOutcomes = predParams.outcomes;
                activeParameters = predParams.parameters;
                if (values != null) {
                    value = values[ci];
                }
                for (int ai = 0; ai < activeOutcomes.length; ai++) {
                    int oid = activeOutcomes[ai];
                    numfeats[oid]++;
                    prior[oid] += activeParameters[ai] * value;
                }
            }
        }

        double normal = 0.0;
        for (int oid = 0; oid < model.getNumOutcomes(); oid++) {
            if (model.getCorrectionParam() != 0) {
                prior[oid] = Math.exp(prior[oid]
                        * model.getConstantInverse()
                        + ((1.0 - ((double) numfeats[oid] / model.getCorrectionConstant())) * model
                                .getCorrectionParam()));
            }
            else {
                prior[oid] = Math.exp(prior[oid] * model.getConstantInverse());
            }
            normal += prior[oid];
        }

        for (int oid = 0; oid < model.getNumOutcomes(); oid++) {
            prior[oid] /= normal;
        }
        return prior;
    }

    public final String getLabelFromIndex(int i) {
        return labels[i];
    }

    public int getNumOutcomes() {
        return (evalParams.getNumOutcomes());
    }

    /**
     * Provides the fundamental data structures which encode the maxent model information. This
     * method will usually only be needed by GISModelWriters. The following values are held in the
     * Object array which is returned by this method:
     * 
     * <li>index 0: opennlp.maxent.Context[] containing the model parameters <li>index 1:
     * java.util.Map containing the mapping of model predicates to unique integers <li>index 2:
     * java.lang.String[] containing the names of the outcomes, stored in the index of the array
     * which represents their unique ids in the model. <li>index 3: java.lang.Integer containing the
     * value of the models correction constant <li>index 4: java.lang.Double containing the value of
     * the models correction parameter
     * 
     * @return An Object[] with the values as described above.
     */
    public final Object[] getDataStructures() {
        Object[] data = new Object[5];
        data[0] = evalParams.getParams();
        data[1] = pmap;
        data[2] = labels;
        data[3] = new Integer((int) evalParams.getCorrectionConstant());
        data[4] = new Double(evalParams.getCorrectionParam());
        return data;
    }

    // public static void main(String[] args) throws java.io.IOException {
    // if (args.length == 0) {
    // System.err.println("Usage: GISModel modelname < contexts");
    // System.exit(1);
    // }
    // GISModel m = new opennlp.maxent.io.SuffixSensitiveGISModelReader(new
    // File(args[0])).getModel();
    // BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
    // DecimalFormat df = new java.text.DecimalFormat(".###");
    // for (String line = in.readLine(); line != null; line = in.readLine()) {
    // String[] context = line.split(" ");
    // double[] dist = m.eval(context);
    // for (int oi=0;oi<dist.length;oi++) {
    // System.out.print("["+m.getOutcome(oi)+" "+df.format(dist[oi])+"] ");
    // }
    // System.out.println();
    // }
    // }
}