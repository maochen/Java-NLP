package org.maochen.nlp.ml.util.dataio;

import com.google.common.collect.ImmutableSet;
import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.classifier.FeatureIndexer;
import org.maochen.nlp.ml.vector.FeatNamedVector;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Created by mguan on 6/28/16.
 */
public class TestCSVDataReader extends CSVDataReader {

    /***
     * @param fields         String line from csv file
     * @param labelCol
     * @param ignoredColumns
     * @param header
     * @param posNegIndex
     * @return tuple may end up with different feat length! Need to go through FeatureIndexer for fixed length one hot representation.
     */
    @Override
    protected Tuple extractValuedFeat(String[] fields, int labelCol, Set<Integer> ignoredColumns, String[] header, int posNegIndex) {
        if (header == null) {
            throw new RuntimeException("This CSV DataReader needs header.");
        }

        Set<String> pipeLineFields = ImmutableSet.of("RelatedActiveMeds01", "RelatedActiveMeds02", "CoveredTextAll", "Location");
        Set<String> catgoricalFeats = ImmutableSet.of("SCUI", "TIME_INSTANCES", "TIME_INTERVALS", "DATE_INSTANCE");
        Set<String> ignoredFeats = ImmutableSet.of("INST_NOTE_DATE", "INST_PARAGRAPH");
        FeatNamedVector featNamedVector = new FeatNamedVector(new double[fields.length - 1 - ignoredColumns.size()]);
        Tuple tuple = new Tuple(featNamedVector);
        tuple.label = fields[labelCol];

        if (posNegIndex > -1) {
            tuple.isPosExample = Integer.parseInt(fields[posNegIndex]) > -1;
        }

        List<String> featName = new ArrayList<>();
        List<Double> values = new ArrayList<>();
        // Iter over fields
        for (int i = 0; i < fields.length; i++) {
            if (i == labelCol || ignoredColumns.contains(i)) {
                continue;
            }

            if (ignoredFeats.contains(header[i])) {
                continue;
            }

            if ("ProbScuiSet".equalsIgnoreCase(header[i])) {
                String[] cell = fields[i].replaceAll("[\\[|\\]]", StringUtils.EMPTY).split(":");
                for (String aCell : cell) {
                    featName.add(header[i] + "_" + aCell.toLowerCase().trim());
                    values.add(1D);
                }
            } else if (pipeLineFields.contains(header[i])) {
                String[] cell = fields[i].split("\\|");
                for (String aCell : cell) {
                    featName.add(header[i] + "_" + aCell.toLowerCase().trim().replaceAll("\\s+", "_"));
                    values.add(1D);
                }
            } else if (catgoricalFeats.contains(header[i])) {
                featName.add(header[i] + "_" + fields[i].toLowerCase().trim().replaceAll("\\s+", "_"));
                values.add(1D);
            } else {
                featName.add(header[i]);
                try {
                    double val = Double.parseDouble(fields[i]);
                    values.add(val);
                } catch (NumberFormatException e) {
                    String newFeatName = featName.get(featName.size() - 1) + "_" + fields[i].toLowerCase().trim().replaceAll("\\s+", "_");
                    featName.set(featName.size() - 1, newFeatName);
                    double val = fields[i].trim().isEmpty() ? 0 : 1;
                    values.add(val);
                }
            }
        }

        ((FeatNamedVector) tuple.vector).featsName = featName.stream().toArray(String[]::new);
        tuple.vector.setVector(values.stream().mapToDouble(i -> i).toArray());
        return tuple;
    }

    public TestCSVDataReader(String filename, int labelCol, String delim, boolean hasHeader, Set<Integer> ignoredColumns, int posNegIndex) {
        super(filename, labelCol, delim, hasHeader, ignoredColumns, posNegIndex);
    }


    public static void main(String[] args) throws IOException {
//        String filename = "/Users/mguan/Desktop/features/feat_batch13_patient3760.tsv";
        String filename = "/Users/mguan/Desktop/test.tsv";
        TestCSVDataReader testCSVDataReader = new TestCSVDataReader(filename, 6, "\t", true, ImmutableSet.of(0, 1, 2, 3, 4, 6), 0);
        List<Tuple> tuples = testCSVDataReader.read();

        FeatureIndexer featureIndexer = new FeatureIndexer();
        double[][] result = featureIndexer.process(tuples.stream().map(x -> (FeatNamedVector) x.vector).collect(Collectors.toList()));
        String[] header = featureIndexer.getFeatNames();
        System.out.println(Arrays.toString(header));
        for (double[] r : result) {
            System.out.println(Arrays.toString(r));
        }

        System.out.println("\n\n\n");


        for (Tuple t : tuples) {
            String[] feats = ((FeatNamedVector) t.vector).featsName;
            double[] val = t.vector.getVector();

            StringBuilder stringBuilder = new StringBuilder();
            for (int i = 0; i < feats.length; i++) {
                stringBuilder.append(feats[i] + "=" + val[i] + ", ");
            }

            System.out.println(stringBuilder.toString());


            System.out.println("----------------");
            System.out.println("");
        }
    }

}
