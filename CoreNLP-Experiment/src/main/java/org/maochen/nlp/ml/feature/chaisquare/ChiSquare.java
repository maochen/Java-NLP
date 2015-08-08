package org.maochen.nlp.ml.feature.chaisquare;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.maochen.nlp.ml.datastructure.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Created by Maochen on 8/7/15.
 */
public class ChiSquare {
    private static final Logger LOG = LoggerFactory.getLogger(ChiSquare.class);

    // http://www.wikihow.com/Calculate-P-Value
    public static final double EMPIRICAL_P_VALUE = 0.05;

    // Row - feats, Col - label
    protected Table<String, String, Integer> dataTable = HashBasedTable.create();

    // Value smaller -> independent
    protected Table<String, String, Double> chiSquareTable = HashBasedTable.create();

    // < EMPIRICAL_P_VALUE -> independent
    protected Table<String, String, Double> pValTable = HashBasedTable.create();

    protected int df;

    protected int total; // N

    public double totalChiSquare;

    public double totalPVal;

    private Function<Table<String, String, Integer>, Integer> getTotal = table1 ->
            table1.rowMap().values().stream().map(Map::values) // List<Collection<Int>>
                    .map(lst -> lst.stream().mapToInt(num -> num).sum()) // List<Int>
                    .mapToInt(num -> num).sum();

    public void loadTrainingData(List<Tuple> trainingData) {

        for (Tuple t : trainingData) {
            String label = t.label;

            for (String featName : t.featureName) {
                Integer count = dataTable.get(featName, label);
                count = count == null ? 1 : count + 1;
                dataTable.put(featName, label, count);
            }
        }

    }

    protected static double getPValue(final double chiSquare, double df) {
        GammaDistribution gamma = new GammaDistribution(df / 2.0D, 2.0D);
        double gammaVal = gamma.cumulativeProbability(chiSquare);

        return 1 - gammaVal;
    }

    public void calculateChiSquare() {
        df = (dataTable.rowKeySet().size() - 1);
        df = df == 0 ? df = 1 : df;
        df = df * (dataTable.columnKeySet().size() - 1);
        df = df == 0 ? df = 1 : df;

        total = getTotal.apply(dataTable);
        // R, C, V
        dataTable.cellSet().forEach(cell -> {
            String feat = cell.getRowKey();
            String label = cell.getColumnKey();
            int count = cell.getValue() == null ? 0 : cell.getValue();
            int c_feat = dataTable.row(feat).values().stream().mapToInt(x -> x).sum();
            int c_label = dataTable.column(label).values().stream().mapToInt(x -> x).sum();
            double e_xi_yi = (c_feat * c_label) / (double) total;
            chiSquareTable.put(feat, label, Math.pow(count - e_xi_yi, 2) / e_xi_yi);
        });

        chiSquareTable.cellSet().forEach(cell -> {
            double pVal = cell.getValue() == null ? 0D : getPValue(cell.getValue(), df);
            pValTable.put(cell.getRowKey(), cell.getColumnKey(), pVal);

        });

        totalChiSquare = chiSquareTable.cellSet().parallelStream().mapToDouble(cell -> cell.getValue() == null ? 0D : cell.getValue()).sum();
        totalPVal = getPValue(totalChiSquare, df);
    }

    public boolean isFeatureUseful() {
        return totalPVal >= EMPIRICAL_P_VALUE;
    }

}
