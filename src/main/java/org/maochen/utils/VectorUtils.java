package org.maochen.utils;

import java.util.Arrays;
import java.util.function.BinaryOperator;
import java.util.stream.IntStream;

/**
 * Created by Maochen on 12/3/14.
 */
public class VectorUtils {

    public static double[] operate(double[] a, double[] b, BinaryOperator<Double> op) {
        if (a.length != b.length) throw new IllegalArgumentException("not same length");
        return IntStream.range(0, a.length).parallel().mapToDouble(i -> op.apply(a[i], b[i])).toArray();
    }

    public static double[] scale(final double[] a, double scale) {
        return Arrays.stream(a).parallel().map(x -> x * scale).toArray();
    }

    public static double gaussianDensityDistribution(double mean, double variance, double x) {
        double twoVariance = 2 * variance;
        double val = 1 / Math.sqrt(Math.PI * twoVariance);
        val = val * Math.exp(-Math.pow((x - mean), 2) / twoVariance);

        return val;
    }

    public static double[] allXVector(double val, int length) {
        return IntStream.range(0, length).parallel().mapToDouble(x -> val).toArray();
    }

    public static float[] doubleToFloat(double[] vector) {
        float[] result = new float[vector.length];
        IntStream.range(0, vector.length).parallel().forEach(i -> result[i] = (float) vector[i]);
        return result;
    }

    public static String[] intToString(int[] vectorIndex) {
        return Arrays.stream(vectorIndex).parallel().<String>mapToObj(String::valueOf).toArray(String[]::new);
    }

}
