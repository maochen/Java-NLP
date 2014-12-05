package org.maochen.utils;

/**
 * Created by Maochen on 12/3/14.
 */
public class VectorUtils {

    public static double[] addition(double[] a, double[] b) {
        if (a.length != b.length) throw new IllegalArgumentException("not same length");

        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }

        return result;
    }

    public static double[] minus(double[] a, double[] b) {
        if (a.length != b.length) throw new IllegalArgumentException("not same length");

        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] - b[i];
        }

        return result;
    }

    public static double[] multiply(double[] a, double[] b) {
        if (a.length != b.length) throw new IllegalArgumentException("not same length");

        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * b[i];
        }

        return result;
    }

    public static double[] scale(double[] a, double scale) {

        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * scale;
        }

        return result;
    }

    public static double gaussianDensityDistribution(double mean, double variance, double x) {
        double twoVariance = 2 * variance;
        double val = 1 / Math.sqrt(Math.PI * twoVariance);
        val = val * Math.exp(-Math.pow((x - mean), 2) / twoVariance);

        return val;
    }

    public static double[] allXVector(double x, int length) {

        double[] result = new double[length];
        for (int i = 0; i < length; i++) {
            result[i] = x;
        }

        return result;
    }

}
