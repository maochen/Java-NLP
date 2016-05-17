package org.maochen.nlp.ml.util.dataio;

import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.vector.FeatNamedVector;
import org.maochen.nlp.ml.vector.IVector;

import java.io.*;
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

    private boolean hasHeader;
    private String[] header;

    public List<Tuple> read() throws IOException {
        FileInputStream fileInputStream = new FileInputStream(filename);
        return read(fileInputStream);
    }

    public List<Tuple> read(InputStream is) throws IOException {
        List<Tuple> ds = new ArrayList<>();

        BufferedReader br = new BufferedReader(new InputStreamReader(is));
        String line = br.readLine();

        int count = 0;
        while (line != null) {
            if (count == 0 && hasHeader) {
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

                    if (hasHeader) {
                        name.add(header[i] + "=" + values[i]);
                    } else {
                        name.add(i + "=" + values[i]);
                    }
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

    public CSVDataReader(String filename, int labelCol, String delim, boolean isRealVal, boolean hasHeader) {
        this.filename = filename;
        this.labelCol = labelCol;
        this.delim = delim;
        this.isRealVal = isRealVal;
        this.hasHeader = hasHeader;
    }
}
