package org.maochen.utils;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Copyright 2014-2015 maochen.org
 * Author: Maochen.G   contact@maochen.org
 * For the detail information about license, check the LICENSE.txt
 * <p>
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * <p>
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * <p>
 * You should have received a copy of the GNU General Public License
 * along with this program ; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA  02111-1307 USA
 * <p>
 * Created by Maochen on 12/3/14.
 */
public class VectorUtils {

    public static double[] zip(final double[] vec1, final double[] vec2, BiFunctionDoublePrimitive op) {
        if (vec1 == null || vec2 == null) return new double[0];
        if (vec1.length != vec2.length) {
            throw new IllegalArgumentException("Two Vectors must has equal length.");
        }

        double[] result = new double[vec1.length];
        IntStream.range(0, vec1.length).parallel().forEach(i -> result[i] = op.apply(vec1[i], vec2[i]));
        return result;
    }

    public static double[] addition(final double[]... vectors) {
        return Arrays.stream(vectors).reduce((vec1, vec2) -> zip(vec1, vec2, (f1, f2) -> f1 + f2)).orElse(null);
    }

    private static double dotProduct(final double[] vec1, final double[] vec2) {
        double[] result = zip(vec1, vec2, (f1, f2) -> f1 * f2);
        return Arrays.stream(result).parallel().sum();
    }

    private static double vectorLen(double[] vector) {
        double len = Arrays.stream(vector).parallel().map(x -> x * x).sum();
        return Math.sqrt(len);
    }

    // cos(theta) = A . B / ||A|| ||B||
    public static double getCosinValue(double[] vector1, double[] vector2) {
        if (vector1 == null || vector2 == null) return 0;
        double dotProduct = VectorUtils.dotProduct(vector1, vector2);
        double euclideanDistance = VectorUtils.vectorLen(vector1) * VectorUtils.vectorLen(vector2);
        double cosineValue = Math.abs(dotProduct / euclideanDistance);
        return cosineValue > 1.0D ? 1.0D : cosineValue;  // because of the precision error
    }

    public static double[] scale(final double[] a, double scale) {
        return Arrays.stream(a).parallel().map(x -> x * scale).toArray();
    }

    public static double gaussianPDF(double mean, double variance, double x) {
        double twoVariance = 2 * variance;
        double probability = 1 / Math.sqrt(Math.PI * twoVariance);
        probability = probability * Math.exp(-Math.pow((x - mean), 2) / twoVariance);

        return probability;
    }

    public static float[] doubleToFloat(double[] vector) {
        float[] result = new float[vector.length];
        IntStream.range(0, vector.length).parallel().forEach(i -> result[i] = (float) vector[i]);
        return result;
    }

    public static double[] floatToDouble(float[] vector) {
        double[] result = new double[vector.length];
        IntStream.range(0, vector.length).parallel().forEach(i -> result[i] = vector[i]);
        return result;
    }

    public static String[] intToString(int[] vectorIndex) {
        return Arrays.stream(vectorIndex).parallel().mapToObj(String::valueOf).toArray(String[]::new);
    }

}
