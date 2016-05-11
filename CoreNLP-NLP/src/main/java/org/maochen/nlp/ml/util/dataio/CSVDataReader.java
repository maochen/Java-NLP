package org.maochen.nlp.ml.util.dataio;

import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.util.TrainingDataUtils;
import org.maochen.nlp.ml.vector.FeatNamedVector;
import org.maochen.nlp.ml.vector.IVector;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by mguan on 4/8/16.
 */
public class CSVDataReader {

    private String filename;
    private int labelCol;
    private String delim;
    private boolean isRealVal;

    private String[] header;

    public List<Tuple> read() throws IOException {
        List<Tuple> ds = new ArrayList<>();

        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
        String line = br.readLine();

        int count = 0;
        while (line != null) {
            if (count == 0) {
                header = line.split(delim);
            } else {
                String[] values = line.split(delim);
                final int actualLabelCol = labelCol == -1 ? values.length - 1 : labelCol;
                String label = values[actualLabelCol];

                List<String> name = new ArrayList<>();
                for (int i = 0; i < values.length; i++) {
                    if (i == actualLabelCol) {
                        continue;
                    }
                    name.add(header[i] + "=" + values[i]);
                }

                IVector v = new FeatNamedVector(name.stream().toArray(String[]::new));
                if (isRealVal) {
                    double[] realV = IntStream.range(0, values.length).filter(i -> i != actualLabelCol)
                            .mapToObj(i -> values[i]).mapToDouble(Double::parseDouble).toArray();
                    v.setVector(realV);
                }
                Tuple t = new Tuple(v);
                t.label = label;
                ds.add(t);
            }

            count++;
            line = br.readLine();
        }

        return ds;
    }

    public String getHeader() {
        return Arrays.stream(this.header).collect(Collectors.joining(delim));
    }

    public CSVDataReader(String filename, int labelCol, String delim, boolean isRealVal) {
        this.filename = filename;
        this.labelCol = labelCol;
        this.delim = delim;
        this.isRealVal = isRealVal;
    }

    public static void main(String[] args) throws IOException {
        String filename = "/Users/mguan/Desktop/train.csv";
        CSVDataReader csvDataReader = new CSVDataReader(filename, -1, ",", true);

        List<Tuple> data = csvDataReader.read();

        List<Tuple> balancedData = TrainingDataUtils.createBalancedTrainingData(data);

        balancedData = balancedData.stream()
                .sorted((t1, t2) -> {
                    int id1 = Integer.parseInt(((FeatNamedVector) (t1.vector)).featsName[0]);
                    int id2 = Integer.parseInt(((FeatNamedVector) (t2.vector)).featsName[0]);
                    return Integer.compare(id1, id2);
                }).collect(Collectors.toList());

        String header = csvDataReader.getHeader();
        CSVDataWriter csvDataWriter = new CSVDataWriter(filename.split("\\.")[0] + ".balanced.csv", -1, ",", header);
        csvDataWriter.excludingCols.add(0); // Bypass write id.
        csvDataWriter.write(balancedData, true, false);
    }
}
