package org.maochen.classifier.perceptron;

import org.maochen.classifier.ModelSerializeUtils;

import java.io.*;
import java.util.Arrays;

/**
 * Created by Maochen on 6/5/15.
 */
public class PerceptronModel {
    double[] weights = null;
    double threshold = 0.5;
    double learningRate = 0.1;

    public void persist(final String filename) {
        try (BufferedWriter output = new BufferedWriter(new FileWriter(new File(filename)))) {
            output.write(String.valueOf(this.threshold));
            output.write(System.lineSeparator());
            output.write(String.valueOf(learningRate));
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
                if (!line.trim().isEmpty()) {
                    if (lineCount == 1) {
                        this.threshold = Double.valueOf(line);
                    } else if (lineCount == 2) {
                        this.learningRate = Double.valueOf(line);
                    } else if (lineCount == 4) {
                        weights = Arrays.stream(line.trim().split("\\s")).mapToDouble(Double::parseDouble).toArray();
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
