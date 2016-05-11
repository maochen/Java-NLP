package org.maochen.nlp.ml.util.dataio;

import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.vector.FeatNamedVector;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by mguan on 4/8/16.
 */
public class CSVDataWriter {
    private String filename;
    private int labelCol;
    private String delim;
    private String header;

    public Set<Integer> excludingCols = new HashSet<>();

    public void write(List<Tuple> trainingData, boolean writeFeatName, boolean writeFeatValue) throws IOException {
        if (!writeFeatName && !writeFeatValue) {
            throw new RuntimeException("At least one of the writeFeatName or writeFeatValue should be true.");
        }

        BufferedWriter output = new BufferedWriter(new FileWriter(new File(filename)));

        StringBuilder stringBuilder = new StringBuilder();
        if (header != null && !header.trim().isEmpty()) {
            stringBuilder.append(header).append(System.lineSeparator());
        }

        int vecLength = trainingData.iterator().next().vector.getVector().length;
        if (labelCol > vecLength - 1 || labelCol < 0) {
            labelCol = vecLength;
        }

        int batchSize = 100;
        for (int countT = 0; countT < trainingData.size(); countT++) {
            Tuple t = trainingData.get(countT);
            for (int i = 0; i < t.vector.getVector().length; i++) {
                if (i == labelCol) {
                    stringBuilder.append(t.label).append(delim);
                }

                if (excludingCols.contains(i)) {
                    continue;
                }

                if (t.vector instanceof FeatNamedVector && writeFeatName) {
                    FeatNamedVector lv = (FeatNamedVector) t.vector;
                    stringBuilder.append(lv.featsName[i]);
                    if (writeFeatValue) {
                        stringBuilder.append("=");
                    }
                }

                if (writeFeatValue) {
                    stringBuilder.append(t.vector.getVector()[i]);
                }

                stringBuilder.append(delim);
            }

            if (labelCol == vecLength) {
                stringBuilder.append(t.label);
            }

            // Delete last if delim.
            if (stringBuilder.substring(stringBuilder.length() - 1, stringBuilder.length()).equals(delim)) {
                stringBuilder.deleteCharAt(stringBuilder.length() - 1);
            }
            stringBuilder.append(System.lineSeparator());
            if (countT % batchSize == 0) {
                output.write(stringBuilder.toString());
                stringBuilder = new StringBuilder();
            }
        }

        output.write(stringBuilder.toString().trim());
        output.close();
    }

    public CSVDataWriter(String filename, int labelCol, String delim, String header) {
        this.filename = filename;
        this.labelCol = labelCol;
        this.delim = delim;
        this.header = header;
    }

}
