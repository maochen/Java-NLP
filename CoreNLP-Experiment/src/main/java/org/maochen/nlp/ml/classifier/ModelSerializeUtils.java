package org.maochen.nlp.ml.classifier;

import org.apache.commons.lang3.StringUtils;

import java.util.Map;
import java.util.Set;

/**
 * Created by Maochen on 6/1/15.
 */
public class ModelSerializeUtils {

    public static String oneDimensionArraySerialize(double[] data) {
        StringBuilder builder = new StringBuilder();
        builder.append(data.length);
        builder.append(System.lineSeparator());

        for (int row = 0; row < data.length; row++) {
            builder.append(data[row]).append(StringUtils.SPACE);
        }
        builder.append(System.lineSeparator());
        return builder.toString();
    }

    public static String twoDimensionalArraySerialize(double[][] data) {
        StringBuilder builder = new StringBuilder();
        builder.append(data.length).append(StringUtils.SPACE).append(data[0].length);
        builder.append(System.lineSeparator());
        for (int row = 0; row < data.length; row++) {
            for (int col = 0; col < data[row].length; col++) {
                builder.append(data[row][col]).append(StringUtils.SPACE);
            }
            builder.append(System.lineSeparator());
        }
        return builder.toString();
    }

    public static <T1, T2> String mapSerialize(Set<Map.Entry<T1, T2>> entries) {
        StringBuilder builder = new StringBuilder();
        for (Map.Entry entry : entries) {
            builder.append(entry.getKey().toString())
                    .append(StringUtils.SPACE)
                    .append(entry.getValue().toString())
                    .append(System.lineSeparator());
        }

        return builder.toString();
    }

}
