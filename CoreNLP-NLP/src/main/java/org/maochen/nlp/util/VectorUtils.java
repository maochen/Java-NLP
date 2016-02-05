package org.maochen.nlp.util;

import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Created by Maochen on 12/3/14.
 */
public class VectorUtils {

    public static double[] zip(final double[] vec1, final double[] vec2, BiFunctionDoublePrimitive op) {
        if (vec1 == null || vec2 == null) {
            return new double[0];
        }
        if (vec1.length != vec2.length) {
            throw new IllegalArgumentException("Two Vectors must has equal length.");
        }

        double[] result = new double[vec1.length];
//        IntStream.range(0, vec1.length).forEach(i -> result[i] = op.apply(vec1[i], vec2[i]));
        for (int i = 0; i < vec1.length; i++) {
            result[i] = op.apply(vec1[i], vec2[i]);
        }
        return result;
    }

    public static double[] addition(final double[]... vectors) {
        int dim = 0;
        for (double[] vector : vectors) {
            if (vector != null && vector.length != 0) {
                dim = vector.length;
                break;
            }
        }

        double[] result = new double[dim];

        for (int i = 0; i < vectors.length; i++) {
            if (vectors[i] == null || vectors[i].length == 0) {
                continue;
            }
            for (int dimension = 0; dimension < vectors[i].length; dimension++) {
                result[dimension] += vectors[i][dimension];
            }
        }

        return result;
//        return Arrays.stream(vectors).filter(x -> x != null && x.length > 0)
//                .reduce((vec1, vec2) -> zip(vec1, vec2, (f1, f2) -> f1 + f2)).orElse(null);
    }

    public static double dotProduct(final double[] vec1, final double[] vec2) {
        double sum = 0;
        for (int i = 0; i < vec1.length; i++) {
            sum += vec1[i] * vec2[i];
        }

        return sum;
//        return Arrays.stream(result).sum();
    }

    public static double vectorLen(double[] vector) {
//        double sum = Arrays.stream(vector).parallel().map(x -> x * x).sum();
        double sum = 0;
        for (double d : vector) {
            sum += d * d;
        }

        return Math.sqrt(sum);
    }

    // cos(theta) = A . B / ||A|| ||B||
    public static double getCosinValue(double[] vector1, double[] vector2) {
        if (vector1 == null || vector2 == null) {
            return 0;
        }
        double dotProduct = VectorUtils.dotProduct(vector1, vector2);
        double euclideanDistance = VectorUtils.vectorLen(vector1) * VectorUtils.vectorLen(vector2);
        double cosineValue = Math.abs(dotProduct / euclideanDistance);
        return cosineValue > 1.0D ? 1.0D : cosineValue;  // because of the precision error
    }

    public static void scale(final double[] a, double scale) {
        for (int i = 0; i < a.length; i++) {
            a[i] = a[i] * scale;
        }
//        return Arrays.stream(a).parallel().map(x -> x * scale).toArray();
    }

    public static double gaussianPDF(double mean, double variance, double x) {
        double twoVariance = 2 * variance;
        double probability = 1 / Math.sqrt(Math.PI * twoVariance);
        probability = probability * Math.exp(-Math.pow((x - mean), 2) / twoVariance);

        return probability;
    }

    public static float[] doubleToFloat(final double[] vector) {
        float[] result = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            result[i] = (float) vector[i];
        }

//        IntStream.range(0, vector.length).parallel().forEach(i -> result[i] = (float) vector[i]);
        return result;
    }

    public static double[] floatToDouble(float[] vector) {
        double[] result = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            result[i] = vector[i];
        }
//        IntStream.range(0, vector.length).parallel().forEach(i -> result[i] = vector[i]);
        return result;
    }

    public static String[] intToString(int[] vectorIndex) {
        String[] result = new String[vectorIndex.length];
        for (int i = 0; i < vectorIndex.length; i++) {
            result[i] = String.valueOf(vectorIndex[i]);
        }
        return result;
//        return Arrays.stream(vectorIndex).parallel().mapToObj(String::valueOf).toArray(String[]::new);
    }

    public static Function<Double, Double> sigmoid = z -> 1 / (1 + Math.exp(-z));   // This is for p(s=1)

    public static Function<Double, Double> tanh = z -> {
        double e2z = Math.exp(2 * z);
        return (e2z - 1) / (e2z + 1);
    };

    public static final BiFunction<double[], double[], Double> euclideanDistance = (v1, v2) -> {
        double result = 0;

        for (int i = 0; i < v1.length; i++) {
            double diff = v1[i] - v2[i];
            diff *= diff;
            result += diff;
        }

        result = Math.sqrt(result);
        return result;
//         Math.sqrt(Arrays.stream(VectorUtils.zip(v1, v2, (x1, y1) -> Math.pow(x1 - y1, 2))).sum());
    };
}
