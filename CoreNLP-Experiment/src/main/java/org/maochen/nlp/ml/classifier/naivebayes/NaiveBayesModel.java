package org.maochen.nlp.ml.classifier.naivebayes;

import com.google.common.collect.Lists;
import org.maochen.nlp.ml.classifier.ModelSerializeUtils;
import org.maochen.nlp.datastructure.LabelIndexer;

import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by Maochen on 5/29/15.
 */
public class NaiveBayesModel {
    // row=labelSize,col=featureLength
    double[][] meanVectors;
    double[][] varianceVectors;

    LabelIndexer labelIndexer;

    Map<Integer, Double> labelPrior;

    public void persist(String filename) {
        try (BufferedWriter output = new BufferedWriter(new FileWriter(new File(filename)))) {
            output.write(ModelSerializeUtils.twoDimensionalArraySerialize(meanVectors));
            output.write(System.lineSeparator());
            output.write(ModelSerializeUtils.twoDimensionalArraySerialize(varianceVectors));
            output.write(System.lineSeparator());

            output.write(ModelSerializeUtils.mapSerialize(labelIndexer.labelIndexer.entrySet()));
            output.write(System.lineSeparator());

            output.write(ModelSerializeUtils.mapSerialize(labelPrior.entrySet()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public void load(InputStream is) {
        labelIndexer = new LabelIndexer(Lists.newArrayList());
        labelPrior = new HashMap<>();

        try (BufferedReader br = new BufferedReader(new InputStreamReader(is))) {
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
                } else if (newItemCount == 2) {
                    labelIndexer.labelIndexer.put(line.split("\\s")[0], Integer.parseInt(line.split("\\s")[1]));
                } else if (newItemCount == 3) {
                    labelPrior.put(Integer.parseInt(line.split("\\s")[0]), Double.parseDouble(line.split("\\s")[1]));
                }

            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
