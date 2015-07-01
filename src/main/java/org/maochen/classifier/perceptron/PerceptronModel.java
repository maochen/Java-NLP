package org.maochen.classifier.perceptron;

import com.google.common.collect.Lists;
import org.maochen.classifier.ModelSerializeUtils;
import org.maochen.datastructure.LabelIndexer;
import org.maochen.datastructure.Tuple;

import java.io.*;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Maochen on 6/5/15.
 */
public class PerceptronModel {
    double learningRate = 0.1;
    double threshold = 0.5;

    double[] bias = null;
    double[][] weights = null;

    LabelIndexer labelIndexer;

    public PerceptronModel() {

    }

    public PerceptronModel(List<Tuple> trainingData) {
        labelIndexer = new LabelIndexer(trainingData);
        int featurelength = trainingData.stream().findFirst().orElse(null).featureVector.length;
        weights = new double[labelIndexer.getLabelSize()][featurelength];
        bias = new double[labelIndexer.getLabelSize()];
        Arrays.stream(weights).forEach(w -> Arrays.stream(w).parallel().forEach(wi -> Math.random())); //init weights
        Arrays.stream(bias).forEach(bi -> Math.random());
    }

    public void persist(final String filename) {
        try (BufferedWriter output = new BufferedWriter(new FileWriter(new File(filename)))) {
            output.write(String.valueOf(learningRate));
            output.write(System.lineSeparator());
            output.write(String.valueOf(threshold));
            output.write(System.lineSeparator());

            output.write(ModelSerializeUtils.oneDimensionArraySerialize(bias));
            output.write(ModelSerializeUtils.twoDimensionalArraySerialize(weights));
            output.write("li" + System.lineSeparator());
            output.write(ModelSerializeUtils.mapSerialize(labelIndexer.labelIndexer.entrySet()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void load(final InputStream is) {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(is))) {
            String line;
            int lineCount = 0;

            boolean isLabelIndexer = false;
            while ((line = br.readLine()) != null) {
                lineCount++;
                line = line.trim();
                if (!line.isEmpty()) {
                    if (lineCount == 1) {
                        this.learningRate = Double.valueOf(line);
                    } else if (lineCount == 2) {
                        this.threshold = Double.valueOf(line);
                    } else if (lineCount == 4) {
                        this.bias = Arrays.stream(line.split("\\s")).mapToDouble(Double::parseDouble).toArray();
                    } else if (lineCount == 5) {
                        int rows = Integer.parseInt(line.split("\\s")[0]);
                        this.weights = new double[rows][];
                        for (lineCount = lineCount + 1; lineCount < rows + 6; lineCount++) {
                            line = br.readLine().trim();
                            this.weights[lineCount - 6] = Arrays.stream(line.split("\\s")).mapToDouble(Double::parseDouble).toArray();
                        }
                    } else if (line.equalsIgnoreCase("li")) {
                        isLabelIndexer = true;
                        this.labelIndexer = new LabelIndexer(Lists.newArrayList());
                    } else if (isLabelIndexer) {
                        this.labelIndexer.labelIndexer.put(line.split("\\s")[0], Integer.parseInt(line.split("\\s")[1]));
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
