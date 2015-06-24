package org.maochen.classifier.perceptron;

import org.maochen.classifier.ModelSerializeUtils;

import java.io.*;
import java.util.Arrays;

/**
 * Created by Maochen on 6/5/15.
 */
public class PerceptronModel {
    double[] weights = null;
    double bias = 0;
    double learningRate = 0.1;
    double threshold = 0.5;
    boolean trainBias = true;

    public PerceptronModel() {

    }

    public PerceptronModel(PerceptronModel model) {
        this.weights = model.weights;
        this.bias = model.bias;
        this.learningRate = model.learningRate;
        this.threshold = model.threshold;
        this.trainBias = model.trainBias;
    }

    public void persist(final String filename) {
        try (BufferedWriter output = new BufferedWriter(new FileWriter(new File(filename)))) {
            output.write(String.valueOf(bias));
            output.write(System.lineSeparator());
            output.write(String.valueOf(learningRate));
            output.write(System.lineSeparator());
            output.write(String.valueOf(threshold));
            output.write(System.lineSeparator());
            output.write(String.valueOf(trainBias));
            output.write(System.lineSeparator());
            output.write(ModelSerializeUtils.oneDimensionArraySerialize(weights));
            output.write(System.lineSeparator());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void load(final InputStream is) {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(is))) {
            String line;
            int lineCount = 0;
            while ((line = br.readLine()) != null) {
                lineCount++;
                line = line.trim();
                if (!line.isEmpty()) {
                    if (lineCount == 1) {
                        this.bias = Double.valueOf(line);
                    } else if (lineCount == 2) {
                        this.learningRate = Double.valueOf(line);
                    } else if (lineCount == 3) {
                        this.threshold = Double.valueOf(line);
                    } else if (lineCount == 4) {
                        this.trainBias = Boolean.valueOf(line);
                    } else if (lineCount == 6) {
                        weights = Arrays.stream(line.trim().split("\\s")).mapToDouble(Double::parseDouble).toArray();
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
