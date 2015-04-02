package org.maochen.utils;

import java.util.Arrays;
import java.util.function.BinaryOperator;
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

    public static double[] zip(double[] a, double[] b, BinaryOperator<Double> op) {
        if (a.length != b.length) throw new IllegalArgumentException("not same length");
        return IntStream.range(0, a.length).parallel().mapToDouble(i -> op.apply(a[i], b[i])).toArray();
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
