package org.maochen.classifier.naivebayes;

import com.google.common.collect.Lists;
import org.apache.commons.lang3.StringUtils;
import org.maochen.datastructure.LabelIndexer;

import java.io.*;
import java.util.Arrays;
import java.util.Map;

/**
 * Created by Maochen on 5/29/15.
 */
public class NaiveBayesModel {
    // row=labelSize,col=featureLength
    double[][] meanVectors;
    double[][] varianceVectors;

    LabelIndexer labelIndexer;

    public void persist(String filename) {
        try (BufferedWriter output = new BufferedWriter(new FileWriter(new File(filename)))) {
            output.write(meanVectors.length + StringUtils.SPACE + meanVectors[0].length);
            output.write(System.lineSeparator());
            for (int row = 0; row < meanVectors.length; row++) {
                StringBuilder builder = new StringBuilder();
                for (int col = 0; col < meanVectors[row].length; col++) {
                    builder.append(meanVectors[row][col]).append(StringUtils.SPACE);
                }
                output.write(builder.toString().trim() + System.lineSeparator());
            }
            output.write(System.lineSeparator());

            output.write(varianceVectors.length + StringUtils.SPACE + varianceVectors[0].length);
            output.write(System.lineSeparator());
            for (int row = 0; row < varianceVectors.length; row++) {
                StringBuilder builder = new StringBuilder();
                for (int col = 0; col < varianceVectors[row].length; col++) {
                    builder.append(varianceVectors[row][col]).append(StringUtils.SPACE);
                }
                output.write(builder.toString().trim() + System.lineSeparator());
            }
            output.write(System.lineSeparator());

            for (Map.Entry<String, Integer> entry : labelIndexer.labelIndexer.entrySet()) {
                output.write(entry.getKey() + StringUtils.SPACE + entry.getValue() + System.lineSeparator());
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public void load(String filename) {
        labelIndexer = new LabelIndexer(Lists.newArrayList());

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;

            int newItemCount = 0;
            boolean isFirstLine = true;
            int row = 0;
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    isFirstLine = true;
                    newItemCount++;
                } else if (newItemCount == 0) {
                    if (isFirstLine) {
                        isFirstLine = false;
                        String[] args = line.split("\\s");
                        meanVectors = new double[Integer.parseInt(args[0])][Integer.parseInt(args[1])];
                        row = 0;
                    } else {
                        String[] values = line.split("\\s");
                        meanVectors[row] = Arrays.stream(values).map(Double::parseDouble).mapToDouble(x -> x).toArray();
                        row++;
                    }
                } else if (newItemCount == 1) {
                    if (isFirstLine) {
                        isFirstLine = false;
                        String[] args = line.split("\\s");
                        varianceVectors = new double[Integer.parseInt(args[0])][Integer.parseInt(args[1])];
                        row = 0;
                    } else {
                        String[] values = line.split("\\s");
                        varianceVectors[row] = Arrays.stream(values).map(Double::parseDouble).mapToDouble(x -> x).toArray();
                        row++;
                    }
                } else {
                    labelIndexer.labelIndexer.put(line.split("\\s")[0], Integer.parseInt(line.split("\\s")[1]));
                }

            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
